"""First-stage retrieval module for BM25 and Query Likelihood.

Handles executing Lucene queries against prebuilt BEIR indexes, formatting
standard TREC run files (top-1000 candidates), enforcing strict score
ordering, and recording execution metadata sidecars.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple, Union

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

logger = logging.getLogger(__name__)


def search(
    searcher: LuceneSearcher,
    query: Union[str, Any],
    k: int = 1000,
) -> List[Tuple[str, float]]:
    """Execute search using LuceneSearcher with either plain text or a Lucene Query object.

    Args:
        searcher: Configured LuceneSearcher instance.
        query: Query string (for F0 / F6) or a Lucene Query object (for F1-F5).
        k: Maximum number of candidate documents to retrieve.

    Returns:
        List of (docid, score) tuples in descending order of retrieval score.
    """
    if isinstance(query, str):
        hits = searcher.search(query, k=k)
    else:
        # Pyserini supports passing Lucene Query objects directly to search()
        hits = searcher.search(query, k=k)

    results: List[Tuple[str, float]] = []
    for hit in hits:
        results.append((str(hit.docid), float(hit.score)))

    return results


def run_first_stage(
    dataset: str,
    retriever: str,
    queries: Dict[str, Union[str, Any]],
    cfg: Dict[str, Any],
    tag: str,
    out_path: Union[str, Path],
) -> Path:
    """Execute first-stage retrieval over all queries and save a standard TREC run file.

    Writes a standard 6-column TREC run file and an accompanying .meta.json sidecar
    recording hyperparameters, timestamps, and per-query latencies.

    Args:
        dataset: Dataset name ('scifact', 'nfcorpus', 'fiqa', 'trec-covid').
        retriever: 'bm25' or 'qld'.
        queries: Mapping of qid string to query string or Lucene Query object.
        cfg: Experiment configuration dictionary.
        tag: Run tag string (e.g. 'bm25_F0', 'qld_F0').
        out_path: Destination path for the .trec file.

    Returns:
        Path to the generated .trec run file.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file = out_file.with_suffix(".meta.json")

    from src.data_loader import get_searcher

    searcher = get_searcher(dataset, retriever, cfg)
    k = int(cfg.get("first_stage_depth", 1000))

    start_utc = datetime.now(timezone.utc)
    start_wall = time.perf_counter()

    query_latencies: Dict[str, float] = {}
    lines_to_write: List[str] = []

    logger.info("Executing %s retrieval on %s (%d queries, top-%d)...", retriever.upper(), dataset, len(queries), k)

    for qid, q_val in tqdm(queries.items(), desc=f"First-stage {tag}", unit="query"):
        t0 = time.perf_counter()
        hits = search(searcher, q_val, k=k)
        t1 = time.perf_counter()
        query_latencies[qid] = t1 - t0

        # Enforce strictly decreasing scores to satisfy TREC invariant (Section 13)
        # In case of exact score ties, subtract an infinitesimal delta to preserve ranking
        prev_score = float("inf")
        for rank_idx, (docid, score) in enumerate(hits, start=1):
            if score >= prev_score:
                score = prev_score - 1e-7
            prev_score = score
            lines_to_write.append(f"{qid} Q0 {docid} {rank_idx} {score:.7f} {tag}\n")

    end_wall = time.perf_counter()
    end_utc = datetime.now(timezone.utc)
    total_seconds = end_wall - start_wall
    avg_latency = total_seconds / max(1, len(queries))

    # Write TREC run file
    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(lines_to_write)
    logger.info("Saved %d run lines to %s", len(lines_to_write), out_file)

    # Gather package versions for reproducibility (Rule 9)
    pkg_versions = {}
    for pkg in ["pyserini", "torch", "sentence_transformers", "ir_measures"]:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = "unknown"

    meta_payload = {
        "dataset": dataset,
        "retriever": retriever,
        "tag": tag,
        "num_queries": len(queries),
        "total_results": len(lines_to_write),
        "first_stage_depth": k,
        "retriever_params": cfg.get(retriever, {}),
        "seed": cfg.get("seed", 42),
        "package_versions": pkg_versions,
        "start_time_utc": start_utc.isoformat(),
        "end_time_utc": end_utc.isoformat(),
        "wall_time_seconds": total_seconds,
        "avg_seconds_per_query": avg_latency,
        "query_latencies": query_latencies,
    }

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    logger.info("Saved metadata sidecar to %s (avg %.4fs/query)", meta_file, avg_latency)

    return out_file
