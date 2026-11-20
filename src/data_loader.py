"""Data loader module for BEIR datasets via Pyserini.

Responsible for loading queries, relevance judgments (qrels), initializing
LuceneSearcher and LuceneIndexReader instances, and extracting raw document text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List

from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher

logger = logging.getLogger(__name__)


def fiqa_sample_qids(cfg: Dict[str, Any]) -> List[str]:
    """Retrieve or generate the fixed deterministic sample of FiQA query IDs.

    Loads from data/fiqa/sample_qids.txt if present; otherwise samples
    fiqa_sample_size query IDs from the FiQA test set using seed 42.

    Args:
        cfg: Experiment configuration dictionary.

    Returns:
        Sorted list of sampled query ID strings.
    """
    data_dir = Path(cfg["paths"]["data"]) / "fiqa"
    sample_file = data_dir / "sample_qids.txt"

    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            sampled_qids = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d FiQA sample query IDs from %s", len(sampled_qids), sample_file)
        return sorted(sampled_qids)

    # File missing: sample deterministically from qrels
    logger.info("Generating FiQA sample query IDs (seed=42, size=%d)...", cfg["fiqa_sample_size"])
    from pyserini.search import get_qrels

    fiqa_topics_key = cfg["datasets"]["fiqa"]["topics"]
    qrels = get_qrels(fiqa_topics_key)
    all_qids = sorted(str(qid) for qid in qrels.keys())

    rng = random.Random(cfg["seed"])
    sampled = rng.sample(all_qids, cfg["fiqa_sample_size"])
    sampled_qids = sorted(sampled)

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_file, "w", encoding="utf-8") as f:
        for qid in sampled_qids:
            f.write(f"{qid}\n")

    logger.info("Saved %d FiQA sample query IDs to %s", len(sampled_qids), sample_file)
    return sampled_qids


def load_queries(dataset: str, cfg: Dict[str, Any]) -> Dict[str, str]:
    """Load query texts for a dataset from Pyserini's prebuilt topics.

    Only queries that have ground-truth qrels are retained. For FiQA,
    queries are further restricted to the fixed sample.

    Args:
        dataset: Dataset identifier ('scifact', 'nfcorpus', 'fiqa', 'trec-covid').
        cfg: Experiment configuration dictionary.

    Returns:
        Mapping of query ID string to query text string, sorted by query ID.
    """
    from pyserini.search import get_qrels, get_topics

    ds_cfg = cfg["datasets"][dataset]
    topics_key = ds_cfg["topics"]

    raw_topics = get_topics(topics_key)
    raw_qrels = get_qrels(topics_key)

    qids_with_qrels = set(str(k) for k in raw_qrels.keys())

    queries: Dict[str, str] = {}
    for qid, topic_data in raw_topics.items():
        qid_str = str(qid)
        if qid_str in qids_with_qrels:
            query_text = topic_data.get("title", "")
            if query_text:
                queries[qid_str] = query_text.strip()

    if dataset == "fiqa":
        sample_qids_set = set(fiqa_sample_qids(cfg))
        queries = {qid: text for qid, text in queries.items() if qid in sample_qids_set}

    # Sort deterministically by qid
    queries = dict(sorted(queries.items(), key=lambda kv: kv[0]))
    logger.info("Loaded %d queries with qrels for dataset '%s'", len(queries), dataset)
    return queries


def load_qrels(dataset: str, cfg: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Load relevance judgments (qrels) for a dataset.

    Args:
        dataset: Dataset identifier ('scifact', 'nfcorpus', 'fiqa', 'trec-covid').
        cfg: Experiment configuration dictionary.

    Returns:
        Nested dict mapping qid -> docid -> relevance grade (as integer).
    """
    from pyserini.search import get_qrels

    ds_cfg = cfg["datasets"][dataset]
    topics_key = ds_cfg["topics"]

    raw_qrels = get_qrels(topics_key)

    qrels: Dict[str, Dict[str, int]] = {}
    for qid, docs in raw_qrels.items():
        qid_str = str(qid)
        qrels[qid_str] = {str(docid): int(rel) for docid, rel in docs.items()}

    if dataset == "fiqa":
        sample_qids_set = set(fiqa_sample_qids(cfg))
        qrels = {qid: docs for qid, docs in qrels.items() if qid in sample_qids_set}

    qrels = dict(sorted(qrels.items(), key=lambda kv: kv[0]))
    logger.info("Loaded qrels for %d queries in dataset '%s'", len(qrels), dataset)
    return qrels


def get_searcher(dataset: str, retriever: str, cfg: Dict[str, Any]) -> LuceneSearcher:
    """Initialize a LuceneSearcher configured with BM25 or Query Likelihood.

    Args:
        dataset: Dataset identifier.
        retriever: 'bm25' or 'qld'.
        cfg: Experiment configuration dictionary.

    Returns:
        Configured LuceneSearcher instance.
    """
    index_name = cfg["datasets"][dataset]["index"]
    searcher = LuceneSearcher.from_prebuilt_index(index_name)

    if retriever == "bm25":
        k1 = float(cfg["bm25"]["k1"])
        b = float(cfg["bm25"]["b"])
        searcher.set_bm25(k1=k1, b=b)
        logger.info("Initialized BM25 searcher for %s (k1=%.2f, b=%.2f)", dataset, k1, b)
    elif retriever == "qld":
        mu = float(cfg["qld"]["mu"])
        searcher.set_qld(mu=mu)
        logger.info("Initialized QLD searcher for %s (mu=%.1f)", dataset, mu)
    else:
        raise ValueError(f"Unknown retriever: {retriever}. Expected 'bm25' or 'qld'.")

    return searcher


def get_index_reader(dataset: str, cfg: Dict[str, Any]) -> LuceneIndexReader:
    """Initialize a LuceneIndexReader for accessing index vocabulary and statistics.

    Args:
        dataset: Dataset identifier.
        cfg: Experiment configuration dictionary.

    Returns:
        LuceneIndexReader instance.
    """
    index_name = cfg["datasets"][dataset]["index"]
    return LuceneIndexReader.from_prebuilt_index(index_name)


def get_doc_text(searcher: LuceneSearcher, docid: str) -> str:
    """Extract and format raw document text (title + ' ' + text) from index.

    Args:
        searcher: LuceneSearcher instance holding the target index.
        docid: Target document ID string.

    Returns:
        Concatenated document title and text string.
    """
    doc = searcher.doc(str(docid))
    if doc is None:
        return ""
    try:
        raw_json = json.loads(doc.raw())
        title = raw_json.get("title", "").strip()
        text = raw_json.get("text", "").strip()
        if title and text:
            return f"{title} {text}"
        return title or text
    except Exception as exc:
        logger.warning("Failed to parse raw JSON for docid %s: %s", docid, exc)
        return doc.contents() or ""
