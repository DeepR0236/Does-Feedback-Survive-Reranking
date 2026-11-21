"""Classical and Generative Pseudo-Relevance Feedback algorithms.

Implements term extraction, vocabulary filtering, vector normalization,
Rocchio (F1, F4), RM3 (F2, F5) with softmax QLD score conversion, HyDE averaging (F3),
naive concatenation (F6), and Lucene BooleanQuery construction.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher
import pyserini.search.lucene.querybuilder as qb

logger = logging.getLogger(__name__)

# Cached default analyzer
_DEFAULT_ANALYZER: Optional[Analyzer] = None


def get_default_analyzer() -> Analyzer:
    """Retrieve or create the standard Lucene analyzer with Porter stemmer and stopwords."""
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = Analyzer(get_lucene_analyzer(stemmer="porter", stopwords=True))
    return _DEFAULT_ANALYZER


def term_vector(text: str, analyzer: Optional[Analyzer] = None) -> Dict[str, float]:
    """Tokenize and stem text to produce term frequency counts.

    Args:
        text: Input raw text string.
        analyzer: Optional Pyserini Analyzer.

    Returns:
        Mapping of stemmed term string to frequency count.
    """
    if analyzer is None:
        analyzer = get_default_analyzer()

    tokens = analyzer.analyze(text)
    counts = Counter(tokens)
    return {t: float(c) for t, c in counts.items()}


def filter_terms(
    vec: Dict[str, float],
    index_reader: LuceneIndexReader,
    cfg: Dict[str, Any],
    alnum_only: bool = False,
) -> Dict[str, float]:
    """Filter vocabulary terms by length, document frequency, and alphanumerics.

    Follows Section 15.1 of specification:
    - Term length between min_term_len and max_term_len (default 2 to 20).
    - df / total_docs <= max_df_ratio (default 0.1).
    - Terms missing from index (df == 0) are dropped.
    - If alnum_only is True (used in RM3 on real documents), non-alphanumeric terms dropped.

    Args:
        vec: Term frequency mapping.
        index_reader: LuceneIndexReader for corpus statistics.
        cfg: Experiment configuration dictionary.
        alnum_only: If True, keep only alphanumeric terms.

    Returns:
        Filtered term frequency mapping.
    """
    fb_cfg = cfg.get("feedback", {})
    min_len = int(fb_cfg.get("min_term_len", 2))
    max_len = int(fb_cfg.get("max_term_len", 20))
    max_df_ratio = float(fb_cfg.get("max_df_ratio", 0.1))

    total_docs = index_reader.stats().get("documents", 0)
    if total_docs <= 0:
        return {}

    filtered: Dict[str, float] = {}
    for term, count in vec.items():
        if not (min_len <= len(term) <= max_len):
            continue
        if alnum_only and not term.isalnum():
            continue

        # Section 15.1: Call get_term_counts on already-analysed terms
        # Use analyzer=None so Pyserini uses stemming=False, stopwords=False and does not crash on stopwords
        try:
            counts = index_reader.get_term_counts(term, analyzer=None)
            doc_freq = counts[0] if counts else 0
        except Exception:
            doc_freq = 0

        if doc_freq is None or doc_freq <= 0:
            continue

        if (doc_freq / total_docs) <= max_df_ratio:
            filtered[term] = float(count)

    return filtered


def corpus_doc_vector(
    docid: str,
    index_reader: LuceneIndexReader,
    cfg: Dict[str, Any],
    alnum_only: bool = False,
) -> Dict[str, float]:
    """Extract and filter the stored term frequency vector of an indexed document.

    Args:
        docid: Target document ID string.
        index_reader: LuceneIndexReader instance.
        cfg: Experiment configuration dictionary.
        alnum_only: Whether to restrict to alphanumeric terms.

    Returns:
        Filtered term frequency mapping.
    """
    raw_vec = index_reader.get_document_vector(str(docid))
    if not raw_vec:
        return {}
    return filter_terms(raw_vec, index_reader, cfg, alnum_only=alnum_only)


def l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    """L2 normalize a term-weight vector (Section 15.1).

    Divides by Euclidean norm if norm > 0.001; returns empty dict if norm <= 0.001.
    """
    if not vec:
        return {}
    sum_sq = sum(w * w for w in vec.values())
    norm = math.sqrt(sum_sq)
    if norm <= 0.001:
        return {}
    return {t: w / norm for t, w in vec.items()}


def l1_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    """L1 normalize a term-weight vector (Section 15.1).

    Divides by sum of absolute values.
    """
    if not vec:
        return {}
    total = sum(abs(w) for w in vec.values())
    if total <= 0:
        return {}
    return {t: w / total for t, w in vec.items()}


def mean_vector(vectors: Sequence[Dict[str, float]], k: int = 128) -> Dict[str, float]:
    """Compute pruned mean vector across a sequence of vectors (Section 15.2).

    1. L2-normalizes each input vector (skipping near-zero vectors).
    2. Averages term weights across vectors.
    3. Retains the top k terms by weight (ties broken alphabetically by term).
    4. L2-normalizes the resulting pruned vector.

    Args:
        vectors: Sequence of term-weight dictionaries.
        k: Maximum number of terms to retain (default 128).

    Returns:
        Pruned and L2-normalized mean term vector.
    """
    normed_vectors = [l2_normalize(v) for v in vectors]
    valid_vectors = [v for v in normed_vectors if v]

    if not valid_vectors:
        return {}

    n = len(valid_vectors)
    all_terms = set()
    for v in valid_vectors:
        all_terms.update(v.keys())

    avg_weights: Dict[str, float] = {}
    for term in all_terms:
        avg_weights[term] = sum(v.get(term, 0.0) for v in valid_vectors) / n

    # Keep top k terms by weight (ties broken alphabetically by term)
    sorted_terms = sorted(avg_weights.items(), key=lambda item: (-item[1], item[0]))
    pruned = dict(sorted_terms[:k])

    return l2_normalize(pruned)


def rocchio(
    query_text: str,
    fb_vectors: Sequence[Dict[str, float]],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """Rocchio relevance feedback (F1 and F4) (Section 15.3).

    Formula:
        w(t) = alpha * q(t) + beta * mean_rel(t)
    where:
        q = L2-normalized query vector
        mean_rel = mean_vector(feedback_vectors, 128)
        alpha = 1.0, beta = 0.75, gamma = 0.0 (no negative feedback)

    Args:
        query_text: Raw query text.
        fb_vectors: Sequence of filtered feedback document/passage vectors.
        cfg: Experiment configuration dictionary.

    Returns:
        Term-weight mapping of positive expansion weights (at most 128 terms).
    """
    fb_cfg = cfg.get("feedback", {})
    rocchio_cfg = fb_cfg.get("rocchio", {})
    alpha = float(rocchio_cfg.get("alpha", 1.0))
    beta = float(rocchio_cfg.get("beta", 0.75))
    max_terms = int(fb_cfg.get("fb_terms", 128))

    q_vec = l2_normalize(term_vector(query_text))
    mean_rel = mean_vector(fb_vectors, k=max_terms)

    combined_terms = set(q_vec.keys()) | set(mean_rel.keys())
    weights: Dict[str, float] = {}

    for term in combined_terms:
        w = alpha * q_vec.get(term, 0.0) + beta * mean_rel.get(term, 0.0)
        if w > 0:
            weights[term] = w

    sorted_items = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    return dict(sorted_items[:max_terms])


def rm3(
    query_text: str,
    fb_vectors: Sequence[Dict[str, float]],
    fb_scores: Sequence[float],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """RM3 relevance model feedback (F2 and F5) (Section 15.4 & Page 20).

    Formula:
        rel(t) = sum_d (weight_d * normalized_tf(t, d))
        w(t) = 0.5 * q(t) + 0.5 * rel(t)
    where:
        q = L1-normalized query vector
        each document vector retains its top 128 terms and is L1-normalized
        fb_scores are pre-normalized to sum to 1.0 (via softmax for QLD)

    Args:
        query_text: Raw query text.
        fb_vectors: Sequence of filtered feedback document/passage vectors.
        fb_scores: Normalized positive weights per feedback document (sum to 1.0).
        cfg: Experiment configuration dictionary.

    Returns:
        Term-weight mapping of positive expansion weights (at most 128 terms).
    """
    fb_cfg = cfg.get("feedback", {})
    rm3_cfg = fb_cfg.get("rm3", {})
    orig_weight = float(rm3_cfg.get("original_query_weight", 0.5))
    exp_weight = 1.0 - orig_weight
    max_terms = int(fb_cfg.get("fb_terms", 128))

    q_vec = l1_normalize(term_vector(query_text))

    # Form relevance model rel(t)
    rel: Dict[str, float] = {}
    for vec, doc_w in zip(fb_vectors, fb_scores):
        if not vec or doc_w <= 0:
            continue
        # Retain top 128 terms per document and L1-normalize
        sorted_doc_terms = sorted(vec.items(), key=lambda item: (-item[1], item[0]))[:max_terms]
        doc_top_vec = l1_normalize(dict(sorted_doc_terms))

        for term, p_t_d in doc_top_vec.items():
            rel[term] = rel.get(term, 0.0) + (doc_w * p_t_d)

    # Prune rel(t) to top 128 terms and L1-normalize
    sorted_rel = sorted(rel.items(), key=lambda item: (-item[1], item[0]))[:max_terms]
    normed_rel = l1_normalize(dict(sorted_rel))

    # Blend original query and relevance model
    all_terms = set(q_vec.keys()) | set(normed_rel.keys())
    final_weights: Dict[str, float] = {}
    for term in all_terms:
        w = orig_weight * q_vec.get(term, 0.0) + exp_weight * normed_rel.get(term, 0.0)
        if w > 0:
            final_weights[term] = w

    sorted_final = sorted(final_weights.items(), key=lambda item: (-item[1], item[0]))[:max_terms]
    return dict(sorted_final)


def hyde_average(
    query_text: str,
    passage_vectors: Sequence[Dict[str, float]],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """HyDE average vector feedback (F3) (Section 15.5).

    Equal-weight mean vector of the original query and N generated passage vectors.
    """
    max_terms = int(cfg.get("feedback", {}).get("fb_terms", 128))
    q_vec = term_vector(query_text)
    combined = [q_vec] + list(passage_vectors)
    return mean_vector(combined, k=max_terms)


def bm25_passage_score(
    query_text: str,
    passage_text: str,
    index_reader: LuceneIndexReader,
    cfg: Dict[str, Any],
) -> float:
    """Calculate BM25 relevance score for a generated passage against a query (Section 14 & 20).

    Uses corpus statistics from index_reader and BM25 hyperparameters (k1, b).
    """
    k1 = float(cfg.get("bm25", {}).get("k1", 0.9))
    b = float(cfg.get("bm25", {}).get("b", 0.4))

    analyzer = get_default_analyzer()
    q_terms = analyzer.analyze(query_text)
    p_terms = analyzer.analyze(passage_text)

    if not q_terms or not p_terms:
        return 0.0

    stats = index_reader.stats()
    num_docs = stats.get("documents", 1)
    total_terms = stats.get("total_terms", 1)
    avgdl = total_terms / max(1, num_docs)

    doc_len = len(p_terms)
    p_tf = Counter(p_terms)
    q_tf = Counter(q_terms)

    score = 0.0
    for term, q_cnt in q_tf.items():
        tf = p_tf.get(term, 0)
        if tf <= 0:
            continue

        try:
            counts = index_reader.get_term_counts(term, analyzer=None)
            doc_freq = counts[0] if counts else 0
        except Exception:
            doc_freq = 0
        if doc_freq is None or doc_freq <= 0:
            continue

        # Lucene/Anserini BM25 IDF formulation
        idf = math.log(1.0 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avgdl)))
        score += idf * tf_norm * q_cnt

    return float(max(0.0, score))


def naive_concat(query_text: str, passages: Sequence[str]) -> str:
    """Naive concatenation feedback (F6) (Section 15.6).

    Passages joined with spaces followed by space and original query.
    If distinct analyzed terms exceeds 900, truncates to avoid Lucene clause limit.
    """
    raw_concat = " ".join(passages) + " " + query_text
    analyzer = get_default_analyzer()
    tokens = analyzer.analyze(raw_concat)

    distinct_tokens = []
    seen = set()
    for t in tokens:
        if t not in seen:
            seen.add(t)
            distinct_tokens.append(t)

    if len(distinct_tokens) > 900:
        logger.warning("Truncating concatenated query to 900 distinct terms to prevent Lucene clause overflow.")
        return " ".join(distinct_tokens[:900])

    return raw_concat


def to_lucene_query(weights: Dict[str, float]):
    """Convert term-to-weight dictionary into a Lucene BooleanQuery with SHOULD clauses (Section 15.7).

    Args:
        weights: Dictionary mapping term strings to positive boost weights.

    Returns:
        Lucene Query object.
    """
    builder = qb.get_boolean_query_builder()
    for term, w in weights.items():
        if w > 0:
            term_query = qb.JTermQuery(qb.JTerm("contents", term))
            boosted = qb.get_boost_query(term_query, float(w))
            builder.add(boosted, qb.JBooleanClauseOccur["should"].value)
    return builder.build()


def build_feedback_queries(
    dataset: str,
    retriever: str,
    condition: str,
    queries: Dict[str, str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct expanded queries for a given dataset, retriever, and feedback condition.

    Persists the term-to-weight mapping (or concatenated text) to
    expansions/expanded_queries/{dataset}_{retriever}_{condition}.json.

    Args:
        dataset: Dataset identifier ('scifact', etc.).
        retriever: 'bm25' or 'qld'.
        condition: 'F1', 'F2', 'F3', 'F4', 'F5', 'F6'.
        queries: Original query dictionary mapping qid -> query_text.
        cfg: Experiment configuration dictionary.

    Returns:
        Dictionary mapping qid to Lucene Query objects (F1-F5) or strings (F6).
    """
    from src.data_loader import get_index_reader, get_searcher

    exp_dir = Path(cfg["paths"]["expansions"]) / "expanded_queries"
    exp_dir.mkdir(parents=True, exist_ok=True)
    cache_path = exp_dir / f"{dataset}_{retriever}_{condition}.json"

    index_reader = get_index_reader(dataset, cfg)
    searcher = get_searcher(dataset, retriever, cfg)
    real_fb_docs = int(cfg.get("feedback", {}).get("real_fb_docs", 8))

    expanded_terms_log: Dict[str, Any] = {}
    query_objects: Dict[str, Any] = {}

    if condition in ("F1", "F2"):
        # Retrieve top real_fb_docs using the target retriever
        logger.info("Generating %s feedback queries for %s (%s)...", condition, dataset, retriever.upper())
        for qid, qtext in queries.items():
            hits = searcher.search(qtext, k=real_fb_docs)
            if not hits:
                # Fallback to original query
                q_vec = term_vector(qtext)
                expanded_terms_log[qid] = q_vec
                query_objects[qid] = to_lucene_query(q_vec)
                continue

            docids = [str(h.docid) for h in hits]
            raw_scores = [float(h.score) for h in hits]

            # Document vectors
            alnum_flag = (condition == "F2")  # Section 15.1: alphanumeric only for RM3 on real docs
            doc_vectors = [corpus_doc_vector(did, index_reader, cfg, alnum_only=alnum_flag) for did in docids]

            if condition == "F1":
                # Rocchio
                weights = rocchio(qtext, doc_vectors, cfg)
            else:
                # RM3: Compute normalized document weights
                if retriever == "qld":
                    # Page 20: Softmax with temperature 1.0 on raw QLD scores
                    shifted = np.array(raw_scores) - np.max(raw_scores)
                    exp_scores = np.exp(shifted)
                    weights_sum = np.sum(exp_scores)
                    fb_weights = (exp_scores / weights_sum).tolist() if weights_sum > 0 else [1.0 / len(raw_scores)] * len(raw_scores)
                else:
                    # BM25: positive scores normalized to sum to 1.0
                    tot_score = sum(max(0.0, s) for s in raw_scores)
                    fb_weights = [max(0.0, s) / tot_score for s in raw_scores] if tot_score > 0 else [1.0 / len(raw_scores)] * len(raw_scores)

                weights = rm3(qtext, doc_vectors, fb_weights, cfg)

            expanded_terms_log[qid] = weights
            query_objects[qid] = to_lucene_query(weights)

    elif condition in ("F3", "F4", "F5", "F6"):
        # Read from HyDE passage cache (Phase 4 integration)
        from src.hyde_generator import load_hyde
        num_passages = int(cfg.get("hyde", {}).get("num_passages", 4))
        hyde_passages = load_hyde(dataset, n=num_passages, cfg=cfg)

        logger.info("Generating %s HyDE feedback queries for %s (%s)...", condition, dataset, retriever.upper())
        for qid, qtext in queries.items():
            passages = hyde_passages.get(qid, [])
            if not passages:
                q_vec = term_vector(qtext)
                expanded_terms_log[qid] = q_vec
                query_objects[qid] = to_lucene_query(q_vec) if condition != "F6" else qtext
                continue

            if condition == "F6":
                # Naive concatenation
                concat_q = naive_concat(qtext, passages)
                expanded_terms_log[qid] = concat_q
                query_objects[qid] = concat_q
                continue

            # Passage vectors
            passage_vecs = [filter_terms(term_vector(p), index_reader, cfg) for p in passages]

            if condition == "F3":
                weights = hyde_average(qtext, passage_vecs, cfg)
            elif condition == "F4":
                weights = rocchio(qtext, passage_vecs, cfg)
            elif condition == "F5":
                # Page 20: BM25 passage scoring with corpus statistics for either retriever
                raw_bm25_scores = [bm25_passage_score(qtext, p, index_reader, cfg) for p in passages]
                tot_bm25 = sum(raw_bm25_scores)
                if tot_bm25 > 0:
                    fb_weights = [s / tot_bm25 for s in raw_bm25_scores]
                else:
                    fb_weights = [1.0 / len(passages)] * len(passages)
                weights = rm3(qtext, passage_vecs, fb_weights, cfg)
            else:
                raise ValueError(f"Unknown condition: {condition}")

            expanded_terms_log[qid] = weights
            query_objects[qid] = to_lucene_query(weights)

    else:
        raise ValueError(f"Unsupported feedback condition: {condition}")

    # Persist expanded queries
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(expanded_terms_log, f, indent=2)
    logger.info("Saved expanded queries to %s", cache_path)

    return query_objects
