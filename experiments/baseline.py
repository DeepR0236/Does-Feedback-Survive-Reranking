#!/usr/bin/env python3
"""Baseline Experiment (F0 - No Feedback Control).

Executes first-stage retrieval for BM25 and Query Likelihood without feedback (F0)
across BEIR datasets, produces TREC run files with top-1000 candidates, evaluates
all standard metrics using ir_measures, records metadata sidecars, and checks
against literature sanity targets (Acceptance Test A2).

Usage:
    python experiments/baseline.py --dataset scifact
    python experiments/baseline.py --dataset all
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import List

# Ensure environment defaults for Java 21 and Pyserini
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.data_loader import load_queries, load_qrels
from src.evaluator import aggregate, evaluate
from src.retriever import run_first_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments.baseline")

# Sanity targets defined in Section 7 of the specification
SANITY_TARGETS = {
    "scifact": 0.679,
    "trec-covid": 0.595,
    "fiqa": 0.236,
}


def run_baseline_for_dataset(
    dataset: str,
    retrievers: List[str],
    cfg: dict,
    force: bool = False,
) -> pd.DataFrame:
    """Run F0 baseline retrieval and evaluation for a given dataset."""
    logger.info("==================================================")
    logger.info(" Processing dataset: %s", dataset.upper())
    logger.info("==================================================")

    queries = load_queries(dataset, cfg)
    qrels = load_qrels(dataset, cfg)
    metrics_list = cfg.get("metrics", ["nDCG@10", "R@100"])

    runs_dir = Path(cfg["paths"]["runs"]) / "first_stage" / dataset
    runs_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(cfg["paths"]["results"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset_per_query_frames: List[pd.DataFrame] = []

    for ret in retrievers:
        tag = f"{ret}_F0"
        run_path = runs_dir / f"{tag}.trec"

        if run_path.exists() and not force:
            logger.info("Run file %s already exists. Skipping retrieval (use --force to re-run).", run_path)
        else:
            logger.info("Running first-stage retrieval for %s...", tag)
            run_first_stage(
                dataset=dataset,
                retriever=ret,
                queries=queries,
                cfg=cfg,
                tag=tag,
                out_path=run_path,
            )

        logger.info("Evaluating %s with ir_measures...", tag)
        eval_df = evaluate(run=run_path, qrels=qrels, metrics=metrics_list)

        # Attach metadata columns matching Section 13 specification
        eval_df["dataset"] = dataset
        eval_df["retriever"] = ret
        eval_df["feedback"] = "F0"
        eval_df["reranked"] = False
        eval_df["depth"] = 0

        # Column order: dataset, retriever, feedback, reranked, depth, qid, metric, value
        cols = ["dataset", "retriever", "feedback", "reranked", "depth", "qid", "metric", "value"]
        eval_df = eval_df[cols]
        dataset_per_query_frames.append(eval_df)

    combined_new_per_query = pd.concat(dataset_per_query_frames, ignore_index=True)

    # Persist or update results/metrics/{dataset}_per_query.csv
    per_query_csv = metrics_dir / f"{dataset}_per_query.csv"
    if per_query_csv.exists() and not force:
        existing_pq = pd.read_csv(per_query_csv)
        # Drop existing F0 rows for these retrievers to update idempotently
        mask = ~(
            (existing_pq["dataset"] == dataset)
            & (existing_pq["feedback"] == "F0")
            & (existing_pq["reranked"] == False)
            & (existing_pq["retriever"].isin(retrievers))
        )
        combined_pq = pd.concat([existing_pq[mask], combined_new_per_query], ignore_index=True)
    else:
        combined_pq = combined_new_per_query

    combined_pq.to_csv(per_query_csv, index=False)
    logger.info("Saved per-query metrics to %s", per_query_csv)

    # Compute and persist aggregate metrics
    agg_df = aggregate(combined_pq)
    aggregate_csv = metrics_dir / f"{dataset}_aggregate.csv"
    agg_df.to_csv(aggregate_csv, index=False)
    logger.info("Saved aggregate metrics to %s", aggregate_csv)

    # Log summary and sanity target comparison
    print("\n" + "=" * 65)
    print(f" BASELINE SUMMARY: {dataset.upper()} (F0 Control)")
    print("=" * 65)
    f0_agg = agg_df[
        (agg_df["dataset"] == dataset)
        & (agg_df["feedback"] == "F0")
        & (agg_df["reranked"] == False)
    ]

    pivoted = f0_agg.pivot(index="metric", columns="retriever", values="value")
    print(pivoted.to_string(float_format=lambda x: f"{x:.4f}"))
    print("-" * 65)

    if "bm25" in retrievers and "nDCG@10" in pivoted.index:
        bm25_ndcg = pivoted.loc["nDCG@10", "bm25"]
        target = SANITY_TARGETS.get(dataset)
        if target is not None:
            diff = abs(bm25_ndcg - target)
            print(f"Sanity Target Comparison (nDCG@10):")
            print(f"  Observed BM25:  {bm25_ndcg:.4f}")
            print(f"  Literature Ref: {target:.4f}")
            print(f"  Delta:          {diff:.4f} (Threshold: ~0.010)")
            if diff <= 0.015:
                print("  --> PASS: Baseline matches sanity target closely!")
            else:
                print("  --> WARNING: Delta exceeds target tolerance. Check prebuilt index version.")
    print("=" * 65 + "\n")

    return combined_pq


def main():
    parser = argparse.ArgumentParser(description="Run baseline retrieval (F0) across datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        help="Dataset name ('scifact', 'nfcorpus', 'fiqa', 'trec-covid', or 'all').",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default=None,
        choices=["bm25", "qld"],
        help="Specific retriever to run (default: all configured in yaml).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of runs even if they exist on disk.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.dataset == "all":
        target_datasets = list(cfg["datasets"].keys())
    else:
        if args.dataset not in cfg["datasets"]:
            raise ValueError(f"Dataset '{args.dataset}' not defined in config.")
        target_datasets = [args.dataset]

    if args.retriever:
        target_retrievers = [args.retriever]
    else:
        target_retrievers = cfg.get("retrievers", ["bm25", "qld"])

    for ds in target_datasets:
        run_baseline_for_dataset(ds, target_retrievers, cfg, force=args.force)


if __name__ == "__main__":
    main()
