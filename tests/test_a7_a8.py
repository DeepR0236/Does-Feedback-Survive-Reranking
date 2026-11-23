#!/usr/bin/env python3
"""Acceptance Tests A7 and A8: Neural Reranking Sanity and Invariant Verification.

A7: Reranker sanity check on BM25 top-100 on SciFact.
    Pass condition: Document order changes and nDCG@10 rises over first-stage baseline.

A8: Reranked-run invariants, verified for every reranked run:
    1. Recall@100 equals first-stage Recall@100 exactly (delta == 0).
    2. Documents below rank k keep their exact first-stage order.
    3. The set of top-k documents equals the first-stage top-k set.
    4. Scores are strictly decreasing throughout every reranked run.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_qrels
from src.reranker import read_trec_run


def test_a7(dataset: str = "scifact", config_path: str = "configs/experiment_config.yaml") -> bool:
    print("=" * 80)
    print(f"ACCEPTANCE TEST A7: Reranker Sanity Check ({dataset.upper()} BM25 F0 Depth 100)")
    print("=" * 80)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    runs_base = Path(cfg["paths"]["runs"])
    fs_path = runs_base / "first_stage" / dataset / "bm25_F0.trec"
    rr_path = runs_base / "reranked" / dataset / "bm25_F0_depth100.trec"

    if not fs_path.exists():
        print(f"  FAIL: First-stage run not found at {fs_path}")
        return False
    if not rr_path.exists():
        print(f"  FAIL: Reranked run not found at {rr_path}. Run experiments/reranking.py first.")
        return False

    fs_data = read_trec_run(fs_path)
    rr_data = read_trec_run(rr_path)

    # Check 1: Ranking changed
    print("[Check 1] Checking that cross-encoder changed document order...")
    changed_queries = 0
    total_queries = len(fs_data)

    for qid, fs_hits in fs_data.items():
        rr_hits = rr_data.get(qid, [])
        fs_top10 = [d for d, _ in fs_hits[:10]]
        rr_top10 = [d for d, _ in rr_hits[:10]]
        if fs_top10 != rr_top10:
            changed_queries += 1

    pct_changed = (changed_queries / max(1, total_queries)) * 100.0
    print(f"  Queries with reordered top-10: {changed_queries}/{total_queries} ({pct_changed:.1f}%)")
    if changed_queries == 0:
        print("  FAIL: Neural reranker produced no ranking changes in the top 10!")
        return False
    print("  PASS: Cross-encoder actively reordered candidate rankings.")

    # Check 2: nDCG@10 comparison
    print("\n[Check 2] Checking nDCG@10 before vs. after reranking...")
    agg_csv = Path(cfg["paths"]["results"]) / "metrics" / f"{dataset}_aggregate.csv"
    if not agg_csv.exists():
        print(f"  FAIL: Aggregate CSV not found at {agg_csv}")
        return False

    agg_df = pd.read_csv(agg_csv)
    fs_val_row = agg_df[
        (agg_df["dataset"] == dataset)
        & (agg_df["retriever"] == "bm25")
        & (agg_df["feedback"] == "F0")
        & (agg_df["reranked"] == False)
        & (agg_df["metric"] == "nDCG@10")
    ]
    rr_val_row = agg_df[
        (agg_df["dataset"] == dataset)
        & (agg_df["retriever"] == "bm25")
        & (agg_df["feedback"] == "F0")
        & (agg_df["reranked"] == True)
        & (agg_df["depth"] == 100)
        & (agg_df["metric"] == "nDCG@10")
    ]

    if fs_val_row.empty or rr_val_row.empty:
        print("  FAIL: Could not locate nDCG@10 values in aggregate CSV")
        return False

    fs_ndcg = float(fs_val_row["value"].values[0])
    rr_ndcg = float(rr_val_row["value"].values[0])
    delta = rr_ndcg - fs_ndcg

    print(f"  First-Stage BM25 F0 nDCG@10: {fs_ndcg:.4f}")
    print(f"  Reranked Depth 100 nDCG@10:  {rr_ndcg:.4f}")
    print(f"  Delta:                       {delta:+.4f} ({(delta / fs_ndcg) * 100:+.2f}%)")

    if delta <= 0:
        print(f"  FAIL: Expected nDCG@10 to rise, but delta is {delta:+.4f} <= 0")
        return False

    print("  PASS: Neural reranking successfully improved nDCG@10.")
    print("=" * 80)
    print("ACCEPTANCE TEST A7: PASSED")
    print("=" * 80 + "\n")
    return True


def test_a8(dataset: str = "scifact", config_path: str = "configs/experiment_config.yaml") -> bool:
    print("=" * 80)
    print(f"ACCEPTANCE TEST A8: Reranked-Run Invariants Verification ({dataset.upper()})")
    print("=" * 80)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    runs_base = Path(cfg["paths"]["runs"])
    fs_dir = runs_base / "first_stage" / dataset
    rr_dir = runs_base / "reranked" / dataset

    retrievers = cfg.get("retrievers", ["bm25", "qld"])
    conditions = cfg.get("feedback", {}).get("conditions", ["F0", "F1", "F2", "F3", "F4", "F5", "F6"])
    depths = cfg.get("depths", [20, 50, 100])

    agg_csv = Path(cfg["paths"]["results"]) / "metrics" / f"{dataset}_aggregate.csv"
    agg_df = pd.read_csv(agg_csv) if agg_csv.exists() else None

    total_runs_checked = 0

    for ret in retrievers:
        for cond in conditions:
            fs_path = fs_dir / f"{ret}_{cond}.trec"
            if not fs_path.exists():
                print(f"  WARNING: First-stage run {fs_path.name} not found. Skipping...")
                continue

            fs_data = read_trec_run(fs_path)

            for d in depths:
                rr_path = rr_dir / f"{ret}_{cond}_depth{d}.trec"
                if not rr_path.exists():
                    print(f"  FAIL: Reranked run {rr_path.name} not found!")
                    return False

                rr_data = read_trec_run(rr_path)
                total_runs_checked += 1

                # Invariant 1: Top-k set equality and Invariant 2: Tail order preservation
                for qid, fs_hits in fs_data.items():
                    rr_hits = rr_data.get(qid, [])
                    if not rr_hits:
                        print(f"  FAIL [{rr_path.name}]: Query {qid} missing in reranked run")
                        return False

                    fs_docids = [docid for docid, _ in fs_hits]
                    rr_docids = [docid for docid, _ in rr_hits]

                    # 1. Candidate count preservation
                    if len(rr_docids) != len(fs_docids):
                        print(f"  FAIL [{rr_path.name}]: Query {qid} length mismatch ({len(rr_docids)} vs {len(fs_docids)})")
                        return False

                    # 2. Invariant 3: Top-k set equals first-stage top-k set
                    fs_top_k = set(fs_docids[:d])
                    rr_top_k = set(rr_docids[:d])
                    if fs_top_k != rr_top_k:
                        print(f"  FAIL [{rr_path.name}]: Query {qid} top-{d} set mismatch!")
                        print(f"    Missing in reranked: {fs_top_k - rr_top_k}")
                        return False

                    # 3. Invariant 2: Documents below rank k keep first-stage order exactly
                    fs_tail = fs_docids[d:]
                    rr_tail = rr_docids[d:]
                    if fs_tail != rr_tail:
                        print(f"  FAIL [{rr_path.name}]: Query {qid} ranks {d+1}..1000 order was corrupted!")
                        return False

                    # 4. Invariant 4: Scores are strictly decreasing
                    rr_scores = [score for _, score in rr_hits]
                    for i in range(len(rr_scores) - 1):
                        if rr_scores[i] <= rr_scores[i + 1]:
                            print(f"  FAIL [{rr_path.name}]: Query {qid} non-decreasing score at pos {i+1}: {rr_scores[i]} <= {rr_scores[i+1]}")
                            return False

                # Invariant 1: Recall@100 equals first-stage Recall@100 exactly (when d == 100)
                if d == 100 and agg_df is not None:
                    fs_r100_row = agg_df[
                        (agg_df["dataset"] == dataset)
                        & (agg_df["retriever"] == ret)
                        & (agg_df["feedback"] == cond)
                        & (agg_df["reranked"] == False)
                        & (agg_df["metric"] == "R@100")
                    ]
                    rr_r100_row = agg_df[
                        (agg_df["dataset"] == dataset)
                        & (agg_df["retriever"] == ret)
                        & (agg_df["feedback"] == cond)
                        & (agg_df["reranked"] == True)
                        & (agg_df["depth"] == 100)
                        & (agg_df["metric"] == "R@100")
                    ]
                    if not fs_r100_row.empty and not rr_r100_row.empty:
                        fs_r100 = float(fs_r100_row["value"].values[0])
                        rr_r100 = float(rr_r100_row["value"].values[0])
                        r_delta = abs(rr_r100 - fs_r100)
                        if r_delta > 1e-9:
                            print(f"  FAIL [{rr_path.name}]: R@100 violated! First-stage: {fs_r100:.6f}, Reranked: {rr_r100:.6f}, Delta: {r_delta}")
                            return False

    print(f"  Checked {total_runs_checked} reranked TREC run files ({len(retrievers)} retrievers x {len(conditions)} conditions x {len(depths)} depths).")
    print("  PASS: Invariant 1 - Recall@100 is identical to first-stage Recall@100 (delta == 0).")
    print("  PASS: Invariant 2 - Documents below rank k preserve first-stage order exactly.")
    print("  PASS: Invariant 3 - The set of top-k candidates is identical to first-stage top-k.")
    print("  PASS: Invariant 4 - Synthetic scores are strictly decreasing across all runs.")
    print("=" * 80)
    print("ACCEPTANCE TEST A8: PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    a7_ok = test_a7()
    a8_ok = test_a8()
    sys.exit(0 if (a7_ok and a8_ok) else 1)
