"""Evaluation module using ir_measures.

Computes standard Information Retrieval metrics across runs, supports oracle
re-ranking ceiling calculation, and aggregates per-query evaluations into
summary reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

import ir_measures
from ir_measures import Qrel, ScoredDoc, parse_measure
import pandas as pd

logger = logging.getLogger(__name__)


def _to_qrel_objects(qrels: Union[str, Path, Dict[str, Dict[str, int]], Sequence[Qrel]]) -> List[Qrel]:
    """Convert various qrels input representations to a list of ir_measures.Qrel objects."""
    if isinstance(qrels, (str, Path)):
        # Load from file using ir_measures reader
        return list(ir_measures.read_trec_qrels(str(qrels)))
    elif isinstance(qrels, dict):
        qrel_list: List[Qrel] = []
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                qrel_list.append(Qrel(query_id=str(qid), doc_id=str(docid), relevance=int(rel)))
        return qrel_list
    elif isinstance(qrels, (list, tuple)):
        return list(qrels)
    else:
        raise TypeError(f"Unsupported qrels format: {type(qrels)}")


def _to_scored_doc_objects(run: Union[str, Path, Sequence[ScoredDoc], pd.DataFrame]) -> List[ScoredDoc]:
    """Convert various run input representations to a list of ir_measures.ScoredDoc objects."""
    if isinstance(run, (str, Path)):
        run_file = Path(run)
        if not run_file.exists():
            raise FileNotFoundError(f"Run file not found: {run_file}")
        return list(ir_measures.read_trec_run(str(run_file)))
    elif isinstance(run, pd.DataFrame):
        docs: List[ScoredDoc] = []
        for _, row in run.iterrows():
            qid = str(row.get("qid") or row.get("query_id") or row[0])
            docid = str(row.get("docid") or row.get("doc_id") or row[2])
            score = float(row.get("score") or row[4])
            docs.append(ScoredDoc(query_id=qid, doc_id=docid, score=score))
        return docs
    elif isinstance(run, (list, tuple)):
        return list(run)
    else:
        raise TypeError(f"Unsupported run format: {type(run)}")


def evaluate(
    run: Union[str, Path, Sequence[ScoredDoc], pd.DataFrame],
    qrels: Union[str, Path, Dict[str, Dict[str, int]], Sequence[Qrel]],
    metrics: Sequence[Union[str, Any]],
) -> pd.DataFrame:
    """Evaluate a run against qrels using ir_measures.

    Ensures that queries with no retrieved results are assigned a score of 0.0
    for every metric, strictly preserving evaluation across all test queries.

    Args:
        run: TREC run file path, list of ScoredDoc, or DataFrame.
        qrels: Qrels file path, nested dict (qid -> docid -> rel), or list of Qrel.
        metrics: Sequence of metric name strings (e.g. 'nDCG@10', 'R@100') or measure objects.

    Returns:
        DataFrame with columns ['qid', 'metric', 'value'].
    """
    qrels_objs = _to_qrel_objects(qrels)
    run_objs = _to_scored_doc_objects(run)

    # Parse metric names into ir_measures objects
    parsed_measures = []
    measure_name_map = {}
    for m in metrics:
        if isinstance(m, str):
            obj = parse_measure(m)
            parsed_measures.append(obj)
            measure_name_map[str(obj)] = m
        else:
            parsed_measures.append(m)
            measure_name_map[str(m)] = str(m)

    # Identify all distinct query IDs present in qrels
    all_qids = sorted(set(q.query_id for q in qrels_objs))

    # Compute per-query metrics
    records: List[Dict[str, Any]] = []
    seen_qid_metrics = set()

    for item in ir_measures.iter_calc(parsed_measures, qrels_objs, run_objs):
        qid = str(item.query_id)
        m_str = measure_name_map.get(str(item.measure), str(item.measure))
        records.append({"qid": qid, "metric": m_str, "value": float(item.value)})
        seen_qid_metrics.add((qid, m_str))

    # Backfill missing queries with 0.0 (Section 14: "Queries with no results score 0; never drop them")
    for qid in all_qids:
        for m in metrics:
            m_str = str(m)
            if (qid, m_str) not in seen_qid_metrics:
                records.append({"qid": qid, "metric": m_str, "value": 0.0})

    df = pd.DataFrame(records)
    df.sort_values(by=["qid", "metric"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def oracle_ndcg10(
    run: Union[str, Path, Sequence[ScoredDoc]],
    qrels: Union[str, Path, Dict[str, Dict[str, int]]],
    depth: int = 100,
) -> pd.DataFrame:
    """Compute the oracle ranking ceiling (nDCG@10) from the top-k candidates.

    Re-orders the top `depth` candidates for each query by their true relevance
    label (highest first), breaking ties by original retrieval order, and scores
    the resulting ideal ordering using nDCG@10.

    Args:
        run: First-stage run file path or ScoredDoc list.
        qrels: Qrels mapping or file.
        depth: Candidate depth to consider (e.g. 20, 50, 100).

    Returns:
        DataFrame with columns ['qid', 'metric', 'value'] where metric is 'oracle_nDCG@10'.
    """
    run_objs = _to_scored_doc_objects(run)
    qrels_objs = _to_qrel_objects(qrels)

    # Build qrels lookup dict: (qid, docid) -> relevance
    qrel_lookup: Dict[Tuple[str, str], int] = {
        (q.query_id, q.doc_id): q.relevance for q in qrels_objs
    }

    # Group run by query
    from collections import defaultdict
    query_docs: Dict[str, List[ScoredDoc]] = defaultdict(list)
    for doc in run_objs:
        query_docs[doc.query_id].append(doc)

    oracle_docs: List[ScoredDoc] = []
    for qid, docs in query_docs.items():
        # Truncate to specified candidate depth
        candidates = docs[:depth]

        # Sort candidates by true relevance (descending), preserving original rank for ties
        # candidates are already in original rank order (0, 1, 2, ...)
        indexed_candidates = list(enumerate(candidates))
        indexed_candidates.sort(
            key=lambda item: (qrel_lookup.get((qid, item[1].doc_id), 0), -item[0]),
            reverse=True,
        )

        # Assign strictly decreasing synthetic scores (e.g. 1000 - rank)
        for rank_idx, (_, c_doc) in enumerate(indexed_candidates, start=1):
            synth_score = float(1000 - rank_idx)
            oracle_docs.append(ScoredDoc(query_id=qid, doc_id=c_doc.doc_id, score=synth_score))

    # Evaluate oracle run with nDCG@10
    oracle_df = evaluate(oracle_docs, qrels_objs, ["nDCG@10"])
    oracle_df["metric"] = "oracle_nDCG@10"
    return oracle_df


def aggregate(per_query: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-query metric evaluations into mean values across queries.

    Args:
        per_query: DataFrame containing per-query results with columns
                   ['dataset', 'retriever', 'feedback', 'reranked', 'depth', 'qid', 'metric', 'value']
                   or ['qid', 'metric', 'value'].

    Returns:
        Aggregated DataFrame with mean metric values.
    """
    # Group by all metadata columns except 'qid' and 'value'
    non_group_cols = {"qid", "value"}
    group_cols = [c for c in per_query.columns if c not in non_group_cols]

    if not group_cols:
        group_cols = ["metric"]

    agg_df = per_query.groupby(group_cols, as_index=False)["value"].mean()
    agg_df.sort_values(by=group_cols, inplace=True)
    agg_df.reset_index(drop=True, inplace=True)
    return agg_df
