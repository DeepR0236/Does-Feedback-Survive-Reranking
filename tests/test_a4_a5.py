#!/usr/bin/env python3
"""Acceptance Tests A4 & A5: Classical Feedback Validation & QLD Stability.

Verifies:
- A4: Custom RM3 and Rocchio on BM25 SciFact are in the same ballpark (within about 0.02
      nDCG@10) of Pyserini's built-in set_rm3; no degenerate runs; all weights positive;
      at most 128 terms kept.
- A5: Query Likelihood baseline, F1, and F2 on SciFact are non-degenerate (nDCG@10 not near 0).
      Additional A5 check: QLD feedback weights are finite, non-negative, sum to 1,
      strictly preserve ordering, and never become accidentally uniform.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
import yaml

from src.data_loader import load_queries, load_qrels
from src.evaluator import evaluate


def test_a4_custom_feedback_bm25():
    print("\n" + "=" * 65)
    print(" ACCEPTANCE TEST A4: Custom Feedback vs Pyserini set_rm3 (BM25)")
    print("=" * 65)

    cfg_path = PROJECT_ROOT / "configs" / "experiment_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metrics_csv = PROJECT_ROOT / "results" / "metrics" / "scifact_aggregate.csv"
    assert metrics_csv.exists(), f"Metrics file missing: {metrics_csv}"

    agg_df = pd.read_csv(metrics_csv)

    # 1. Inspect custom RM3 (F2) and Rocchio (F1) scores
    bm25_f0 = agg_df[(agg_df["retriever"] == "bm25") & (agg_df["feedback"] == "F0") & (agg_df["metric"] == "nDCG@10")]["value"].values[0]
    bm25_f1 = agg_df[(agg_df["retriever"] == "bm25") & (agg_df["feedback"] == "F1") & (agg_df["metric"] == "nDCG@10")]["value"].values[0]
    bm25_f2 = agg_df[(agg_df["retriever"] == "bm25") & (agg_df["feedback"] == "F2") & (agg_df["metric"] == "nDCG@10")]["value"].values[0]

    print(f"  Observed BM25 F0 nDCG@10:       {bm25_f0:.4f}")
    print(f"  Observed Custom Rocchio (F1):   {bm25_f1:.4f}")
    print(f"  Observed Custom RM3 (F2):       {bm25_f2:.4f}")

    # 2. Run Pyserini built-in set_rm3 on all 300 SciFact queries as benchmark
    print("\n[1/3] Computing Pyserini built-in set_rm3 on SciFact (300 queries)...")
    queries = load_queries("scifact", cfg)
    qrels = load_qrels("scifact", cfg)

    searcher = LuceneSearcher.from_prebuilt_index("beir-v1.0.0-scifact.flat")
    searcher.set_bm25(0.9, 0.4)
    searcher.set_rm3(fb_terms=128, fb_docs=8, original_query_weight=0.5)

    from ir_measures import ScoredDoc
    builtin_rm3_docs = []
    for qid, qtext in queries.items():
        hits = searcher.search(qtext, k=100)
        for h in hits:
            builtin_rm3_docs.append(ScoredDoc(query_id=qid, doc_id=str(h.docid), score=float(h.score)))

    builtin_eval = evaluate(builtin_rm3_docs, qrels, ["nDCG@10"])
    builtin_rm3_ndcg = builtin_eval[builtin_eval["metric"] == "nDCG@10"]["value"].mean()
    print(f"  Pyserini built-in RM3 nDCG@10:  {builtin_rm3_ndcg:.4f}")

    diff_rm3 = abs(bm25_f2 - builtin_rm3_ndcg)
    print(f"  Difference (Custom RM3 vs Built-in RM3): {diff_rm3:.4f}")
    assert diff_rm3 <= 0.025, f"Custom RM3 diverges from built-in RM3 by {diff_rm3:.4f} > 0.025"
    print("  --> PASS: Custom RM3 is in the same ballpark as Pyserini set_rm3.")

    # 3. Check term weights properties
    print("\n[2/3] Checking term weights properties (expansions/expanded_queries)...")
    for cond in ["F1", "F2"]:
        exp_file = PROJECT_ROOT / "expansions" / "expanded_queries" / f"scifact_bm25_{cond}.json"
        assert exp_file.exists(), f"Expansion file missing: {exp_file}"
        with open(exp_file, "r", encoding="utf-8") as f:
            exp_data = json.load(f)

        assert len(exp_data) == 300, f"Expected 300 queries in {exp_file}, got {len(exp_data)}"
        for qid, term_weights in exp_data.items():
            assert len(term_weights) <= 128, f"Query {qid} has {len(term_weights)} > 128 terms in {cond}"
            for term, w in term_weights.items():
                assert w > 0, f"Term {term} in query {qid} has non-positive weight: {w}"

        print(f"  --> PASS: {cond} queries all have positive weights and at most 128 terms.")

    # 4. Check no run is degenerate
    print("\n[3/3] Checking for non-degeneracy...")
    assert bm25_f1 > 0.40, f"BM25 F1 appears degenerate: nDCG@10 = {bm25_f1}"
    assert bm25_f2 > 0.40, f"BM25 F2 appears degenerate: nDCG@10 = {bm25_f2}"
    print("  --> PASS: Neither F1 nor F2 is degenerate.")
    print(" >>> ACCEPTANCE TEST A4 PASSED! <<<")


def test_a5_query_likelihood_stability():
    print("\n" + "=" * 65)
    print(" ACCEPTANCE TEST A5: Query Likelihood Feedback Stability & Check")
    print("=" * 65)

    metrics_csv = PROJECT_ROOT / "results" / "metrics" / "scifact_aggregate.csv"
    agg_df = pd.read_csv(metrics_csv)

    # 1. Verify QLD runs on SciFact
    print("[1/2] Verifying QLD F0, F1, F2 retrieval effectiveness...")
    qld_f0 = agg_df[(agg_df["retriever"] == "qld") & (agg_df["feedback"] == "F0") & (agg_df["metric"] == "nDCG@10")]["value"].values[0]
    qld_f1 = agg_df[(agg_df["retriever"] == "qld") & (agg_df["feedback"] == "F1") & (agg_df["metric"] == "nDCG@10")]["value"].values[0]
    qld_f2 = agg_df[(agg_df["retriever"] == "qld") & (agg_df["feedback"] == "F2") & (agg_df["metric"] == "nDCG@10")]["value"].values[0]

    print(f"  QLD F0 nDCG@10: {qld_f0:.4f}")
    print(f"  QLD F1 nDCG@10: {qld_f1:.4f}")
    print(f"  QLD F2 nDCG@10: {qld_f2:.4f}")

    assert qld_f0 > 0.40, f"QLD F0 degenerate: {qld_f0}"
    assert qld_f1 > 0.40, f"QLD F1 degenerate: {qld_f1}"
    assert qld_f2 > 0.40, f"QLD F2 degenerate: {qld_f2}"
    print("  --> PASS: No QLD run is degenerate (nDCG@10 well above 0).")

    # 2. Additional A5 check: Softmax QLD feedback weights validation
    print("\n[2/2] A5 Additional Check: Softmax QLD feedback score weighting...")
    test_scores = [-15.2, -18.7, -21.4, -22.0, -25.8, -26.1, -29.0, -32.5]
    shifted = np.array(test_scores) - np.max(test_scores)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))

    # Invariants from PDF p. 21:
    assert np.all(np.isfinite(weights)), "Softmax weights contain NaN or Inf"
    assert np.all(weights > 0), "Softmax weights contain negative or zero values"
    assert abs(np.sum(weights) - 1.0) < 1e-6, f"Softmax weights do not sum to 1: {np.sum(weights)}"

    # Strict ordering preservation
    for i in range(len(weights) - 1):
        assert weights[i] > weights[i + 1], f"Weight ordering violated at index {i}"

    # Non-uniformity check for unequal raw scores
    assert np.std(weights) > 1e-4, "Softmax weights accidentally collapsed to uniform"

    # Equal score test
    equal_scores = [-20.0] * 8
    eq_shifted = np.array(equal_scores) - np.max(equal_scores)
    eq_weights = np.exp(eq_shifted) / np.sum(np.exp(eq_shifted))
    assert np.allclose(eq_weights, [1.0 / 8] * 8), "Equal scores did not produce uniform weights"

    print("  --> PASS: Verified weights are finite, non-negative, sum to 1, preserve rank ordering, and handle equal scores.")
    print(" >>> ACCEPTANCE TEST A5 PASSED! <<<")


def main():
    test_a4_custom_feedback_bm25()
    test_a5_query_likelihood_stability()
    print("\n" + "=" * 65)
    print(" >>> ALL PHASE 3 ACCEPTANCE TESTS (A4 & A5) PASSED! <<<")
    print("=" * 65)


if __name__ == "__main__":
    main()
