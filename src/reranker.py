"""Neural Cross-Encoder Reranker module with SQLite Score Caching.

Implements candidate reranking using cross-encoder/ms-marco-MiniLM-L-6-v2,
persistent SQLite caching of (qid, docid) scores, depth truncation (20, 50, 100),
synthetic score generation, and ranking invariant enforcement (Acceptance Tests A7 & A8).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Cached CrossEncoder instance
_CROSS_ENCODER_MODEL = None


def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length: int = 512):
    """Load or retrieve the cached CrossEncoder instance on GPU/CPU."""
    global _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_MODEL is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CrossEncoder model '%s' on %s (max_length=%d)...", model_name, device, max_length)
        _CROSS_ENCODER_MODEL = CrossEncoder(model_name, max_length=max_length, device=device)
    return _CROSS_ENCODER_MODEL


def get_db_connection(cfg: Dict[str, Any]) -> sqlite3.Connection:
    """Open or initialize the SQLite score cache database."""
    cache_dir = Path(cfg["paths"]["cache"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "rerank_scores.sqlite"

    conn = sqlite3.connect(str(db_path), timeout=60.0)
    # Enable WAL mode for fast concurrent reads and robust transactions
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                dataset TEXT,
                model TEXT,
                qid TEXT,
                docid TEXT,
                score REAL,
                PRIMARY KEY (dataset, model, qid, docid)
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_lookup ON scores(dataset, model, qid, docid);")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS throughput (
                dataset TEXT,
                model TEXT,
                n_pairs INTEGER,
                seconds REAL,
                batch_size INTEGER,
                created_utc TEXT
            );
        """)
    return conn


def read_trec_run(run_path: Union[str, Path]) -> Dict[str, List[Tuple[str, float]]]:
    """Parse a standard 6-column TREC run file.

    Format: qid Q0 docid rank score tag

    Returns:
        Mapping of qid to list of (docid, score) in file order.
    """
    run_file = Path(run_path)
    if not run_file.exists():
        raise FileNotFoundError(f"Run file not found: {run_file}")

    run_dict: Dict[str, List[Tuple[str, float]]] = {}
    with open(run_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, _, score_str = parts[:5]
                if qid not in run_dict:
                    run_dict[qid] = []
                run_dict[qid].append((docid, float(score_str)))
    return run_dict


def score_pairs(
    dataset: str,
    run: Union[str, Path, Dict[str, List[Tuple[str, float]]]],
    queries: Dict[str, str],
    doc_text_fn: Callable[[Any, str], str],
    searcher: Any,
    cfg: Dict[str, Any],
    top_candidates_depth: int = 100,
) -> int:
    """Score missing (qid, docid) pairs for top candidates and populate SQLite cache.

    Section 14:
    - Load CrossEncoder once.
    - Query text is ALWAYS the original query (never expanded).
    - Document text is title plus text from doc_text_fn.
    - Score only pairs missing from the cache in batches.
    - Record throughput in the throughput table.

    Args:
        dataset: Target dataset name.
        run: Path to TREC run file or parsed run dictionary.
        queries: Mapping of qid to original query string.
        doc_text_fn: Function (searcher, docid) -> doc_text string.
        searcher: LuceneSearcher or index access object.
        cfg: Experiment configuration dictionary.
        top_candidates_depth: Candidate depth to cache (default 100).

    Returns:
        Number of newly scored pairs.
    """
    if isinstance(run, (str, Path)):
        run_data = read_trec_run(run)
    else:
        run_data = run

    rerank_cfg = cfg.get("reranker", {})
    model_name = rerank_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_length = int(rerank_cfg.get("max_length", 512))
    batch_size = int(rerank_cfg.get("batch_size", 32))

    conn = get_db_connection(cfg)

    # 1. Identify all candidate pairs up to top_candidates_depth
    all_needed_pairs: List[Tuple[str, str]] = []
    for qid, hits in run_data.items():
        if qid not in queries:
            continue
        top_hits = hits[:top_candidates_depth]
        for docid, _ in top_hits:
            all_needed_pairs.append((str(qid), str(docid)))

    # Deduplicate needed pairs
    unique_pairs = list(dict.fromkeys(all_needed_pairs))
    if not unique_pairs:
        return 0

    # 2. Check which pairs already exist in cache
    cursor = conn.cursor()
    # Batch check in SQLite using temporary table or chunked queries
    cursor.execute(
        "SELECT qid, docid FROM scores WHERE dataset = ? AND model = ?",
        (dataset, model_name),
    )
    cached_set: Set[Tuple[str, str]] = set((row[0], row[1]) for row in cursor.fetchall())

    missing_pairs = [pair for pair in unique_pairs if pair not in cached_set]

    if not missing_pairs:
        logger.debug("All %d pairs for %s are already cached in SQLite.", len(unique_pairs), dataset)
        return 0

    logger.info(
        "Found %d missing pairs to score for %s (%s, batch_size=%d)...",
        len(missing_pairs),
        dataset,
        model_name,
        batch_size,
    )

    # 3. Fetch texts for missing pairs
    model = get_cross_encoder(model_name, max_length=max_length)
    input_pairs = []
    valid_missing_pairs = []

    for qid, docid in missing_pairs:
        q_text = queries[qid]
        d_text = doc_text_fn(searcher, docid)
        input_pairs.append([q_text, d_text])
        valid_missing_pairs.append((qid, docid))

    # 4. Run CrossEncoder inference in batches
    t0 = time.perf_counter()
    scores = model.predict(
        input_pairs,
        batch_size=batch_size,
        show_progress_bar=len(input_pairs) > 100,
    )
    t1 = time.perf_counter()
    elapsed = max(1e-4, t1 - t0)

    # 5. Insert new scores into SQLite in a single transaction
    rows_to_insert = [
        (dataset, model_name, qid, docid, float(score))
        for (qid, docid), score in zip(valid_missing_pairs, scores)
    ]

    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO scores (dataset, model, qid, docid, score) VALUES (?, ?, ?, ?, ?)",
            rows_to_insert,
        )
        conn.execute(
            "INSERT INTO throughput (dataset, model, n_pairs, seconds, batch_size, created_utc) VALUES (?, ?, ?, ?, ?, ?)",
            (dataset, model_name, len(rows_to_insert), elapsed, batch_size, datetime.now(timezone.utc).isoformat()),
        )

    throughput = len(rows_to_insert) / elapsed
    logger.info(
        "Scored and cached %d pairs in %.2fs (%.1f pairs/sec).",
        len(rows_to_insert),
        elapsed,
        throughput,
    )
    return len(rows_to_insert)


def get_scores(dataset: str, qid: str, docids: Sequence[str], cfg: Dict[str, Any]) -> Dict[str, float]:
    """Retrieve cached neural scores for a query and list of docids.

    Args:
        dataset: Target dataset name.
        qid: Query ID string.
        docids: Sequence of candidate docid strings.
        cfg: Experiment configuration dictionary.

    Returns:
        Mapping of docid -> neural score.
    """
    rerank_cfg = cfg.get("reranker", {})
    model_name = rerank_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    conn = get_db_connection(cfg)
    cursor = conn.cursor()

    # Query matching records
    placeholders = ",".join("?" for _ in docids)
    query_sql = f"""
        SELECT docid, score FROM scores
        WHERE dataset = ? AND model = ? AND qid = ? AND docid IN ({placeholders})
    """
    params = [dataset, model_name, str(qid)] + [str(d) for d in docids]
    cursor.execute(query_sql, params)

    return {row[0]: float(row[1]) for row in cursor.fetchall()}


def rerank_run(
    first_stage_hits: Sequence[Tuple[str, float]],
    scores: Dict[str, float],
    depth: int,
) -> List[Tuple[str, float]]:
    """Rerank candidates up to specified depth, preserving ranks depth+1..1000 in first-stage order.

    Section 5.4 & 14 Invariants:
    1. Sort the top `depth` documents by cross-encoder score descending.
       Ties broken by first-stage rank (lower original index comes first).
    2. Ranks depth+1 to 1000 remain in their exact first-stage order.
    3. Assigns strictly decreasing synthetic scores: 1000 - position.

    Args:
        first_stage_hits: List of (docid, first_stage_score) from initial retrieval.
        scores: Mapping of docid -> neural score.
        depth: Truncation depth for neural reranking (e.g. 20, 50, 100).

    Returns:
        List of (docid, synthetic_score) for all candidates.
    """
    if not first_stage_hits:
        return []

    top_k = first_stage_hits[:depth]
    tail = first_stage_hits[depth:]

    # Stable sort on top_k:
    # Python sort is stable; sorting by score descending automatically preserves
    # first-stage order for identical scores.
    def get_sort_key(item: Tuple[int, Tuple[str, float]]) -> Tuple[float, int]:
        orig_idx, (docid, _) = item
        # If score is missing from cache, default to negative infinity
        s = scores.get(docid, float("-inf"))
        return (s, -orig_idx)

    indexed_top_k = list(enumerate(top_k))
    indexed_top_k.sort(key=get_sort_key, reverse=True)

    reranked_top_k = [item[1][0] for item in indexed_top_k]
    tail_docids = [item[0] for item in tail]

    full_docids = reranked_top_k + tail_docids

    # Assign strictly decreasing synthetic scores: (1000 - position)
    # Position starts at 1, so rank 1 gets 999.0, rank 2 gets 998.0, etc.
    reranked_results: List[Tuple[str, float]] = []
    for pos, docid in enumerate(full_docids, start=1):
        synthetic_score = float(1000.0 - pos)
        reranked_results.append((docid, synthetic_score))

    return reranked_results


def write_reranked_trec_run(
    reranked_data: Dict[str, List[Tuple[str, float]]],
    out_path: Union[str, Path],
    tag: str,
    meta_payload: Dict[str, Any],
) -> Path:
    """Write standard 6-column TREC run file and metadata sidecar for a reranked run."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file = out_file.with_suffix(".meta.json")

    lines = []
    for qid, hits in reranked_data.items():
        for rank, (docid, score) in enumerate(hits, start=1):
            lines.append(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    return out_file
