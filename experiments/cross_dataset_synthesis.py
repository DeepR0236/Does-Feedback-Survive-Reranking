#!/usr/bin/env python3
"""Cross-Dataset Synthesis and Milestone M2 Evaluation Engine.

Aggregates results across all 4 benchmark datasets (SciFact, NFCorpus, FiQA, TREC-COVID),
computes cross-dataset macro-averages, builds unified cross-domain transfer tables,
and generates multi-dataset publication figures.

Artifacts Generated:
- results/metrics/cross_dataset_T1_macro_results.csv
- results/metrics/cross_dataset_T2_transfer_synthesis.csv
- results/metrics/cross_dataset_T3_significance_summary.csv
- results/plots/cross_dataset_Figure_1_ndcg10_compression.png
- results/plots/cross_dataset_Figure_2_depth_curves.png
- results/plots/cross_dataset_Figure_3_recall_transfer_scatter.png
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cross_dataset_synthesis")

DATASETS = ["scifact", "nfcorpus", "fiqa", "trec-covid"]
FEEDBACK_CONDS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
RETRIEVERS = ["bm25", "qld"]


def build_macro_results_table(cfg: dict) -> pd.DataFrame:
    """Build Table T1 Macro Results: Mean metric values across all 4 datasets."""
    metrics_dir = Path(cfg["paths"]["results"]) / "metrics"
    all_aggs = []

    for ds in DATASETS:
        agg_file = metrics_dir / f"{ds}_aggregate.csv"
        if not agg_file.exists():
            raise FileNotFoundError(f"Missing aggregate file: {agg_file}")
        df = pd.read_csv(agg_file)
        all_aggs.append(df)

    combined = pd.concat(all_aggs, ignore_index=True)

    # Compute macro-average across datasets for each (retriever, feedback, reranked, depth, metric)
    group_cols = ["retriever", "feedback", "reranked", "depth", "metric"]
    macro_df = combined.groupby(group_cols, as_index=False)["value"].mean()
    macro_df.insert(0, "dataset", "macro_average")

    out_file = metrics_dir / "cross_dataset_T1_macro_results.csv"
    macro_df.to_csv(out_file, index=False)
    logger.info("Saved cross-dataset macro results to %s", out_file)
    return macro_df


def build_transfer_synthesis_table(cfg: dict) -> pd.DataFrame:
    """Build Table T2 Transfer Synthesis: Combines T2 from all 4 datasets."""
    metrics_dir = Path(cfg["paths"]["results"]) / "metrics"
    t2_frames = []

    for ds in DATASETS:
        t2_file = metrics_dir / f"{ds}_T2_recall_ndcg_transfer.csv"
        if t2_file.exists():
            df = pd.read_csv(t2_file)
            t2_frames.append(df)

    combined_t2 = pd.concat(t2_frames, ignore_index=True)
    out_file = metrics_dir / "cross_dataset_T2_transfer_synthesis.csv"
    combined_t2.to_csv(out_file, index=False)
    logger.info("Saved cross-dataset transfer synthesis to %s", out_file)
    return combined_t2


def build_significance_summary_table(cfg: dict) -> pd.DataFrame:
    """Build Table T3 Significance Summary across all 4 datasets."""
    sig_dir = Path(cfg["paths"]["results"]) / "significance"
    sig_frames = []

    for ds in DATASETS:
        sig_file = sig_dir / f"{ds}_planned.csv"
        if sig_file.exists():
            df = pd.read_csv(sig_file)
            sig_frames.append(df)

    combined_sig = pd.concat(sig_frames, ignore_index=True)
    out_file = Path(cfg["paths"]["results"]) / "metrics" / "cross_dataset_T3_significance_summary.csv"
    combined_sig.to_csv(out_file, index=False)
    logger.info("Saved cross-dataset significance summary to %s", out_file)
    return combined_sig


def generate_cross_dataset_plots(cfg: dict, macro_df: pd.DataFrame, t2_df: pd.DataFrame):
    """Generate multi-dataset publication figures."""
    plots_dir = Path(cfg["paths"]["results"]) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(cfg["paths"]["results"]) / "metrics"

    # -------------------------------------------------------------
    # Figure 1: Spread Compression across all 4 datasets (First-Stage vs Depth 100)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        agg_file = metrics_dir / f"{ds}_aggregate.csv"
        df = pd.read_csv(agg_file)

        # Focus on BM25 nDCG@10
        fs_df = df[(df["retriever"] == "bm25") & (df["reranked"] == False) & (df["metric"] == "nDCG@10")].set_index("feedback")["value"]
        rr_df = df[(df["retriever"] == "bm25") & (df["reranked"] == True) & (df["depth"] == 100) & (df["metric"] == "nDCG@10")].set_index("feedback")["value"]

        x = np.arange(len(FEEDBACK_CONDS))
        width = 0.35

        fs_vals = [fs_df.get(c, 0.0) for c in FEEDBACK_CONDS]
        rr_vals = [rr_df.get(c, 0.0) for c in FEEDBACK_CONDS]

        fs_spread = max(fs_vals) - min(fs_vals)
        rr_spread = max(rr_vals) - min(rr_vals)
        compression = (1.0 - (rr_spread / max(1e-9, fs_spread))) * 100

        ax.bar(x - width/2, fs_vals, width, label="First-Stage (No Rerank)", color="#2b5c8f", alpha=0.9)
        ax.bar(x + width/2, rr_vals, width, label="Reranked Depth 100", color="#d95f02", alpha=0.9)

        ax.set_title(f"{ds.upper()} (Spread: {fs_spread:.4f} → {rr_spread:.4f}, {compression:.1f}% comp.)", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(FEEDBACK_CONDS, fontsize=10)
        ax.set_ylabel("nDCG@10", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="lower right" if ds != "fiqa" else "upper left", fontsize=8)

    plt.suptitle("Cross-Dataset Evaluation: Method Spread Compression (BM25)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig1_path = plots_dir / "cross_dataset_Figure_1_ndcg10_compression.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info("Saved %s", fig1_path)

    # -------------------------------------------------------------
    # Figure 2: Reranking Depth Curves across Datasets (F0 vs F3 vs F4)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)

    depth_labels = ["Depth 0", "Depth 20", "Depth 50", "Depth 100"]

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        agg_file = metrics_dir / f"{ds}_aggregate.csv"
        df = pd.read_csv(agg_file)

        for cond, col, marker in [("F0", "#333333", "o"), ("F3", "#2b5c8f", "s"), ("F4", "#d95f02", "^")]:
            vals = []
            # Depth 0
            d0_row = df[(df["retriever"] == "bm25") & (df["feedback"] == cond) & (df["reranked"] == False) & (df["metric"] == "nDCG@10")]
            vals.append(float(d0_row["value"].values[0]) if not d0_row.empty else 0.0)

            for d in [20, 50, 100]:
                dr_row = df[(df["retriever"] == "bm25") & (df["feedback"] == cond) & (df["reranked"] == True) & (df["depth"] == d) & (df["metric"] == "nDCG@10")]
                vals.append(float(dr_row["value"].values[0]) if not dr_row.empty else 0.0)

            ax.plot(range(4), vals, label=cond, marker=marker, color=col, linewidth=2)

        ax.set_title(ds.upper(), fontsize=12, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels(depth_labels, fontsize=9)
        ax.set_ylabel("nDCG@10", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.suptitle("Impact of Neural Reranking Candidate Depth on nDCG@10", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig2_path = plots_dir / "cross_dataset_Figure_2_depth_curves.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info("Saved %s", fig2_path)

    # -------------------------------------------------------------
    # Figure 3: Scatter Plot of Candidate Recall vs. Reranked nDCG@10
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6.5))
    markers = {"scifact": "o", "nfcorpus": "s", "fiqa": "^", "trec-covid": "D"}
    colors = {
        "F0": "#7f7f7f", "F1": "#1f77b4", "F2": "#aec7e8",
        "F3": "#ff7f0e", "F4": "#2ca02c", "F5": "#98df8a", "F6": "#d62728"
    }

    for _, row in t2_df[t2_df["retriever"] == "bm25"].iterrows():
        ds = row["dataset"]
        cond = row["feedback"]
        r100 = row["first_stage_R@100"]
        ndcg = row["reranked_nDCG@10"]

        plt.scatter(
            r100, ndcg,
            marker=markers.get(ds, "o"),
            color=colors.get(cond, "#333333"),
            s=90,
            alpha=0.85,
            edgecolors="k",
            linewidth=0.5
        )

    # Create custom legends for dataset shapes and feedback colors
    ds_handles = [plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="#555555", markersize=9, label=d.upper()) for d, m in markers.items()]
    cond_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=cond) for cond, c in colors.items()]

    leg1 = plt.legend(handles=ds_handles, title="Dataset", loc="lower right", framealpha=0.9)
    plt.gca().add_artist(leg1)
    plt.legend(handles=cond_handles, title="Feedback Strategy", loc="upper left", framealpha=0.9)

    plt.title("Candidate Recall@100 vs. Reranked nDCG@10 across All Datasets (BM25)", fontsize=12, fontweight="bold")
    plt.xlabel("First-Stage Candidate Recall@100", fontsize=11)
    plt.ylabel("Reranked nDCG@10 (Depth 100)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = plots_dir / "cross_dataset_Figure_3_recall_transfer_scatter.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info("Saved %s", fig3_path)


def main():
    with open("configs/experiment_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.info("=================================================================")
    logger.info(" EXECUTING CROSS-DATASET SYNTHESIS & MILESTONE M2 EVALUATION")
    logger.info("=================================================================")

    macro_df = build_macro_results_table(cfg)
    t2_df = build_transfer_synthesis_table(cfg)
    sig_df = build_significance_summary_table(cfg)
    generate_cross_dataset_plots(cfg, macro_df, t2_df)

    # Print Macro-Average Table
    print("\n" + "=" * 90)
    print(" CROSS-DATASET MACRO-AVERAGE RESULTS (nDCG@10 Across All 4 Datasets)")
    print("=" * 90)

    for stage_label, reranked_flag in [("First-Stage (No Rerank)", False), ("Reranked Depth 100", True)]:
        sub = macro_df[(macro_df["metric"] == "nDCG@10") & (macro_df["reranked"] == reranked_flag)]
        if reranked_flag:
            sub = sub[sub["depth"] == 100]
        p = sub.pivot(index="retriever", columns="feedback", values="value")[FEEDBACK_CONDS]
        print(f"\n--- {stage_label} ---")
        print(p.round(4))

    print("\n" + "=" * 90)
    print(" MILESTONE M2 EVALUATION & CROSS-DATASET SYNTHESIS COMPLETE!")
    print("=" * 90)


if __name__ == "__main__":
    main()
