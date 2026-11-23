#!/usr/bin/env python3
"""Neural Reranking Experiment (Depths 20, 50, 100).

Applies the Cross-Encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) to all
14 first-stage runs (F0–F6 for BM25 and QLD) on a dataset, utilizing an SQLite
score cache (cache/rerank_scores.sqlite) to ensure zero duplicate inference.

Generates reranked TREC runs with synthetic scores, validates ranking invariants,
evaluates all 12 standard ir_measures metrics across candidate depths (20, 50, 100),
and updates results/metrics/{dataset}_per_query.csv and aggregate CSV.

Usage:
    python experiments/reranking.py --dataset scifact
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm
import yaml

from src.data_loader import get_doc_text, get_searcher, load_qrels, load_queries
from src.evaluator import aggregate, evaluate
from src.reranker import get_scores, read_trec_run, rerank_run, score_pairs, write_reranked_trec_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments.reranking")


def run_reranking_experiment(
    dataset: str,
    cfg: dict,
    retrievers: List[str] = None,
    conditions: List[str] = None,
    depths: List[int] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Execute neural reranking at depths 20, 50, 100 across all first-stage runs."""
    if retrievers is None:
        retrievers = cfg.get("retrievers", ["bm25", "qld"])
    if conditions is None:
        conditions = cfg.get("feedback", {}).get("conditions", ["F0", "F1", "F2", "F3", "F4", "F5", "F6"])
    if depths is None:
        depths = cfg.get("depths", [20, 50, 100])

    logger.info("==================================================")
    logger.info(" Neural Reranking Experiment: %s", dataset.upper())
    logger.info(" Retrievers: %s | Conditions: %s | Depths: %s", retrievers, conditions, depths)
    logger.info("==================================================")

    # 1. Load ground truth and searcher
    queries = load_queries(dataset, cfg)
    qrels = load_qrels(dataset, cfg)
    metrics_list = cfg.get("metrics", ["nDCG@10", "nDCG@20", "R@20", "R@50", "R@100"])

    searcher = get_searcher(dataset, "bm25", cfg)

    first_stage_dir = Path(cfg["paths"]["runs"]) / "first_stage" / dataset
    reranked_dir = Path(cfg["paths"]["runs"]) / "reranked" / dataset
    reranked_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(cfg["paths"]["results"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 2. Phase 5 Step 1: Pre-score top 100 candidates for all 14 runs into SQLite cache
    logger.info("\n--- Step 1: Populating SQLite Score Cache for Top 100 Candidates ---")
    max_depth = max(depths)
    total_newly_scored = 0

    for ret in retrievers:
        for cond in conditions:
            tag = f"{ret}_{cond}"
            first_stage_run = first_stage_dir / f"{tag}.trec"
            if not first_stage_run.exists():
                logger.warning("First-stage run not found: %s. Skipping...", first_stage_run)
                continue

            n_scored = score_pairs(
                dataset=dataset,
                run=first_stage_run,
                queries=queries,
                doc_text_fn=get_doc_text,
                searcher=searcher,
                cfg=cfg,
                top_candidates_depth=max_depth,
            )
            total_newly_scored += n_scored

    logger.info("Step 1 complete. Total new pairs scored on GPU: %d.", total_newly_scored)

    # Gather package versions for sidecars
    pkg_versions = {}
    for pkg in ["torch", "sentence_transformers", "ir_measures", "pyserini"]:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = "unknown"

    # 3. Phase 5 Step 2: Generate reranked runs and evaluate at each depth
    logger.info("\n--- Step 2: Generating Reranked Runs & Evaluating ---")
    new_per_query_frames: List[pd.DataFrame] = []

    for ret in retrievers:
        for cond in conditions:
            fs_tag = f"{ret}_{cond}"
            first_stage_run_path = first_stage_dir / f"{fs_tag}.trec"
            if not first_stage_run_path.exists():
                continue

            fs_run_data = read_trec_run(first_stage_run_path)

            for d in depths:
                rerank_tag = f"{ret}_{cond}_depth{d}"
                rerank_run_path = reranked_dir / f"{rerank_tag}.trec"

                if rerank_run_path.exists() and not force:
                    logger.info("Reranked run %s exists. Skipping generation.", rerank_run_path.name)
                else:
                    t0 = time.perf_counter()
                    reranked_dict: Dict[str, List[Tuple[str, float]]] = {}

                    for qid, hits in fs_run_data.items():
                        # Candidates up to depth d
                        top_docids = [docid for docid, _ in hits[:d]]
                        scores_map = get_scores(dataset, qid, top_docids, cfg)
                        reranked_hits = rerank_run(hits, scores_map, depth=d)
                        reranked_dict[qid] = reranked_hits

                    t1 = time.perf_counter()
                    elapsed = t1 - t0

                    meta_payload = {
                        "dataset": dataset,
                        "retriever": ret,
                        "feedback": cond,
                        "reranked": True,
                        "depth": d,
                        "tag": rerank_tag,
                        "model": cfg.get("reranker", {}).get("model"),
                        "num_queries": len(reranked_dict),
                        "package_versions": pkg_versions,
                        "elapsed_seconds": elapsed,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                    }

                    write_reranked_trec_run(reranked_dict, rerank_run_path, rerank_tag, meta_payload)
                    logger.info("Generated %s in %.3fs", rerank_tag, elapsed)

                # Evaluate with ir_measures
                eval_df = evaluate(run=rerank_run_path, qrels=qrels, metrics=metrics_list)
                eval_df["dataset"] = dataset
                eval_df["retriever"] = ret
                eval_df["feedback"] = cond
                eval_df["reranked"] = True
                eval_df["depth"] = d

                cols = ["dataset", "retriever", "feedback", "reranked", "depth", "qid", "metric", "value"]
                eval_df = eval_df[cols]
                new_per_query_frames.append(eval_df)

    if not new_per_query_frames:
        logger.warning("No reranked runs evaluated.")
        return pd.DataFrame()

    combined_reranked = pd.concat(new_per_query_frames, ignore_index=True)

    # 4. Upsert into results/metrics/{dataset}_per_query.csv
    per_query_csv = metrics_dir / f"{dataset}_per_query.csv"
    if per_query_csv.exists() and not force:
        existing_pq = pd.read_csv(per_query_csv)
        # Drop matching reranked records to update idempotently
        mask = ~(
            (existing_pq["dataset"] == dataset)
            & (existing_pq["reranked"] == True)
            & (existing_pq["feedback"].isin(conditions))
            & (existing_pq["retriever"].isin(retrievers))
            & (existing_pq["depth"].isin(depths))
        )
        combined_pq = pd.concat([existing_pq[mask], combined_reranked], ignore_index=True)
    else:
        combined_pq = combined_reranked

    combined_pq.to_csv(per_query_csv, index=False)
    logger.info("Saved updated per-query metrics to %s", per_query_csv)

    # 5. Compute and persist updated aggregate metrics
    agg_df = aggregate(combined_pq)
    aggregate_csv = metrics_dir / f"{dataset}_aggregate.csv"
    agg_df.to_csv(aggregate_csv, index=False)
    logger.info("Saved updated aggregate metrics to %s", aggregate_csv)

    # 6. Milestone M1: Print Before vs. After Reranking Comparison Tables
    print("\n" + "=" * 90)
    print(f" MILESTONE M1: {dataset.upper()} BEFORE VS. AFTER RERANKING (nDCG@10)")
    print("=" * 90)

    ndcg_agg = agg_df[(agg_df["dataset"] == dataset) & (agg_df["metric"] == "nDCG@10")].copy()
    ndcg_agg["depth_label"] = ndcg_agg.apply(
        lambda r: "No Rerank" if not r["reranked"] else f"Depth {int(r['depth'])}", axis=1
    )

    for ret in retrievers:
        print(f"\n--- Retriever: {ret.upper()} (nDCG@10 across Feedback Conditions & Depths) ---")
        ret_df = ndcg_agg[ndcg_agg["retriever"] == ret]
        piv = ret_df.pivot(index="depth_label", columns="feedback", values="value")
        # Order rows: No Rerank, Depth 20, Depth 50, Depth 100
        row_order = [r for r in ["No Rerank", "Depth 20", "Depth 50", "Depth 100"] if r in piv.index]
        piv = piv.reindex(row_order)
        col_order = [c for c in ["F0", "F1", "F2", "F3", "F4", "F5", "F6"] if c in piv.columns]
        piv = piv[col_order]
        print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 90)
    print(f" MILESTONE M1: {dataset.upper()} CANDIDATE COVERAGE (Recall@100)")
    print("=" * 90)
    r100_agg = agg_df[(agg_df["dataset"] == dataset) & (agg_df["metric"] == "R@100") & (agg_df["reranked"] == False)]
    r100_piv = r100_agg.pivot(index="retriever", columns="feedback", values="value")
    col_order = [c for c in ["F0", "F1", "F2", "F3", "F4", "F5", "F6"] if c in r100_piv.columns]
    print(r100_piv[col_order].to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 90 + "\n")

    return combined_pq


def main():
    parser = argparse.ArgumentParser(description="Execute neural reranking experiments.")
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset name.")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to config.")
    parser.add_argument("--force", action="store_true", help="Force re-generation of reranked runs.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_reranking_experiment(dataset=args.dataset, cfg=cfg, force=args.force)


if __name__ == "__main__":
    main()
