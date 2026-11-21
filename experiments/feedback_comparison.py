#!/usr/bin/env python3
"""Feedback Comparison Experiment (F1–F6 First-Stage Retrieval).

Executes first-stage retrieval with Pseudo-Relevance Feedback (F1 Rocchio, F2 RM3,
F3 HyDE average, F4 HyDE+Rocchio, F5 HyDE+RM3, F6 naive concatenation) across BEIR
datasets, saves 6-column TREC run files with top-1000 candidates, evaluates standard
metrics using ir_measures, records metadata sidecars, and updates per-query and
aggregate CSV reports.

Usage:
    python experiments/feedback_comparison.py --dataset scifact --conditions F1 F2
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.data_loader import load_queries, load_qrels
from src.evaluator import aggregate, evaluate
from src.feedback import build_feedback_queries
from src.retriever import run_first_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments.feedback_comparison")


def run_feedback_comparison(
    dataset: str,
    conditions: List[str],
    retrievers: List[str],
    cfg: dict,
    force: bool = False,
) -> pd.DataFrame:
    """Run feedback generation, retrieval, and evaluation for specified conditions and retrievers."""
    logger.info("==================================================")
    logger.info(" Feedback Comparison: %s (%s)", dataset.upper(), ", ".join(conditions))
    logger.info("==================================================")

    queries = load_queries(dataset, cfg)
    qrels = load_qrels(dataset, cfg)
    metrics_list = cfg.get("metrics", ["nDCG@10", "R@100"])

    runs_dir = Path(cfg["paths"]["runs"]) / "first_stage" / dataset
    runs_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(cfg["paths"]["results"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    new_per_query_frames: List[pd.DataFrame] = []

    for cond in conditions:
        for ret in retrievers:
            tag = f"{ret}_{cond}"
            run_path = runs_dir / f"{tag}.trec"

            if run_path.exists() and not force:
                logger.info("Run file %s already exists. Skipping retrieval (use --force to re-run).", run_path)
            else:
                logger.info("Building feedback queries for %s...", tag)
                fb_queries = build_feedback_queries(dataset, ret, cond, queries, cfg)

                logger.info("Running first-stage retrieval for %s...", tag)
                run_first_stage(
                    dataset=dataset,
                    retriever=ret,
                    queries=fb_queries,
                    cfg=cfg,
                    tag=tag,
                    out_path=run_path,
                )

            logger.info("Evaluating %s with ir_measures...", tag)
            eval_df = evaluate(run=run_path, qrels=qrels, metrics=metrics_list)

            eval_df["dataset"] = dataset
            eval_df["retriever"] = ret
            eval_df["feedback"] = cond
            eval_df["reranked"] = False
            eval_df["depth"] = 0

            cols = ["dataset", "retriever", "feedback", "reranked", "depth", "qid", "metric", "value"]
            eval_df = eval_df[cols]
            new_per_query_frames.append(eval_df)

    combined_new = pd.concat(new_per_query_frames, ignore_index=True)

    # Upsert into results/metrics/{dataset}_per_query.csv
    per_query_csv = metrics_dir / f"{dataset}_per_query.csv"
    if per_query_csv.exists() and not force:
        existing_pq = pd.read_csv(per_query_csv)
        # Drop matching (retriever, condition) combinations to update idempotently
        mask = ~(
            (existing_pq["dataset"] == dataset)
            & (existing_pq["reranked"] == False)
            & (existing_pq["feedback"].isin(conditions))
            & (existing_pq["retriever"].isin(retrievers))
        )
        combined_pq = pd.concat([existing_pq[mask], combined_new], ignore_index=True)
    else:
        combined_pq = combined_new

    combined_pq.to_csv(per_query_csv, index=False)
    logger.info("Saved per-query metrics to %s", per_query_csv)

    # Compute and persist aggregate metrics
    agg_df = aggregate(combined_pq)
    aggregate_csv = metrics_dir / f"{dataset}_aggregate.csv"
    agg_df.to_csv(aggregate_csv, index=False)
    logger.info("Saved aggregate metrics to %s", aggregate_csv)

    # Print summary comparison table
    print("\n" + "=" * 75)
    print(f" FEEDBACK COMPARISON SUMMARY: {dataset.upper()}")
    print("=" * 75)
    target_agg = agg_df[
        (agg_df["dataset"] == dataset)
        & (agg_df["reranked"] == False)
        & (agg_df["feedback"].isin(["F0"] + conditions))
    ]

    for ret in retrievers:
        print(f"\n--- Retriever: {ret.upper()} ---")
        ret_df = target_agg[target_agg["retriever"] == ret]
        pivoted = ret_df.pivot(index="metric", columns="feedback", values="value")
        print(pivoted.to_string(float_format=lambda x: f"{x:.4f}"))

    print("=" * 75 + "\n")
    return combined_pq


def main():
    parser = argparse.ArgumentParser(description="Run feedback comparison experiments (F1-F6).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="scifact",
        help="Dataset name ('scifact', 'nfcorpus', 'fiqa', 'trec-covid', or 'all').",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["F1", "F2"],
        help="Feedback conditions to evaluate (e.g. F1 F2).",
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
        run_feedback_comparison(ds, args.conditions, target_retrievers, cfg, force=args.force)


if __name__ == "__main__":
    main()
