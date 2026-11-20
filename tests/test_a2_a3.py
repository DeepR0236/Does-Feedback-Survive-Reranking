#!/usr/bin/env python3
"""Acceptance Tests A2 & A3: Baseline Sanity & Evaluator Exactness.

Verifies:
- A2: BM25 baseline (F0) nDCG@10 on SciFact (300 queries) is within 0.01 of reference 0.679.
- A3: Evaluator correctness: ir_measures nDCG@10 agrees with official trec_eval to 1e-4.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

# Ensure environment defaults
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.data_loader import load_qrels
from src.evaluator import evaluate


def run_trec_eval(qrels_key: str, run_path: Path) -> float:
    """Run pyserini.eval.trec_eval and extract the aggregate ndcg_cut_10 value."""
    cmd = [
        sys.executable,
        "-m",
        "pyserini.eval.trec_eval",
        "-c",
        "-m",
        "ndcg_cut.10",
        qrels_key,
        str(run_path),
    ]
    env = os.environ.copy()
    env["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    output = proc.stdout

    # Look for line: ndcg_cut_10 \t all \t 0.6789
    match = re.search(r"ndcg_cut_10\s+all\s+([0-9\.]+)", output)
    if not match:
        raise ValueError(f"Could not parse ndcg_cut_10 from trec_eval output:\n{output}")
    return float(match.group(1))


def test_a2_scifact_baseline():
    print("\n" + "=" * 60)
    print(" ACCEPTANCE TEST A2: BM25 Baseline Sanity (SciFact)")
    print("=" * 60)

    cfg_path = PROJECT_ROOT / "configs" / "experiment_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_path = PROJECT_ROOT / "runs" / "first_stage" / "scifact" / "bm25_F0.trec"
    assert run_path.exists(), f"Run file missing: {run_path}. Run experiments/baseline.py first."

    # Validate TREC format invariants
    print("[1/3] Validating TREC run format and invariants...")
    with open(run_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) > 0, "Run file is empty"
    query_line_counts = {}
    prev_qid = None
    prev_score = float("inf")
    prev_rank = 0

    for line_idx, line in enumerate(lines, start=1):
        parts = line.strip().split()
        assert len(parts) == 6, f"Line {line_idx} does not have 6 columns: {line}"
        qid, q0, docid, rank_str, score_str, tag = parts
        assert q0 == "Q0", f"Line {line_idx}: expected Q0, got {q0}"
        rank = int(rank_str)
        score = float(score_str)

        if qid != prev_qid:
            prev_qid = qid
            prev_score = float("inf")
            prev_rank = 0
            query_line_counts[qid] = 0

        query_line_counts[qid] += 1
        assert rank == prev_rank + 1, f"Line {line_idx}: rank is {rank}, expected {prev_rank + 1}"
        assert score < prev_score, f"Line {line_idx}: score {score} is not strictly less than prev {prev_score}"
        prev_score = score
        prev_rank = rank

    print(f"  --> PASS: Verified {len(lines)} lines across {len(query_line_counts)} queries.")
    print("  --> Invariant PASS: Scores are strictly decreasing; ranks sequential starting at 1.")

    # Evaluate with ir_measures
    print("\n[2/3] Evaluating BM25 F0 with ir_measures...")
    qrels = load_qrels("scifact", cfg)
    df = evaluate(run=run_path, qrels=qrels, metrics=["nDCG@10", "R@100"])

    ndcg10_mean = df[df["metric"] == "nDCG@10"]["value"].mean()
    r100_mean = df[df["metric"] == "R@100"]["value"].mean()
    target_ndcg = 0.679
    delta = abs(ndcg10_mean - target_ndcg)

    print(f"  SciFact BM25 F0 observed nDCG@10: {ndcg10_mean:.4f}")
    print(f"  SciFact BM25 F0 observed R@100:   {r100_mean:.4f}")
    print(f"  Literature target nDCG@10:        {target_ndcg:.4f}")
    print(f"  Absolute difference:              {delta:.4f}")

    assert delta <= 0.015, f"Observed nDCG@10 {ndcg10_mean:.4f} deviates too much from target {target_ndcg}"
    print(f"  --> PASS: Difference {delta:.4f} is within tolerance (<= 0.015).")
    return ndcg10_mean


def test_a3_evaluator_exactness(ir_ndcg10: float):
    print("\n" + "=" * 60)
    print(" ACCEPTANCE TEST A3: Evaluator Exactness vs. trec_eval")
    print("=" * 60)

    cfg_path = PROJECT_ROOT / "configs" / "experiment_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    topics_key = cfg["datasets"]["scifact"]["topics"]
    run_path = PROJECT_ROOT / "runs" / "first_stage" / "scifact" / "bm25_F0.trec"

    print("Running official trec_eval via pyserini.eval.trec_eval...")
    trec_eval_ndcg = run_trec_eval(topics_key, run_path)

    print(f"  ir_measures nDCG@10: {ir_ndcg10:.6f}")
    print(f"  trec_eval   nDCG@10: {trec_eval_ndcg:.6f}")

    diff = abs(ir_ndcg10 - trec_eval_ndcg)
    print(f"  Absolute difference: {diff:.6e} (Threshold: 1e-4 = 0.0001)")

    assert diff <= 1e-4, f"ir_measures and trec_eval differ by {diff} > 1e-4"
    print("  --> PASS: ir_measures and trec_eval agree to within 1e-4!")


def main():
    ndcg10 = test_a2_scifact_baseline()
    test_a3_evaluator_exactness(ndcg10)
    print("\n" + "=" * 60)
    print(" >>> ACCEPTANCE TESTS A2 & A3 PASSED! <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()
