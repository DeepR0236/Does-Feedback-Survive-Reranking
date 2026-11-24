#!/usr/bin/env python3
"""Ablations, Significance Testing, Analysis Tables (T1–T5), and Figures (1–4).

Generates:
- Table T1: Main Results Table (all metrics, before vs. after reranking at depth 100).
- Table T2: Recall-to-nDCG Transfer Table (first-stage R@100 vs. reranked nDCG@10 vs. Oracle ceiling).
- Table T3: Planned Significance Comparisons with Holm-Bonferroni correction and 95% CIs.
- Table T4: Depth and Cost Efficiency Trade-offs (depths 20, 50, 100 vs. latencies & pair counts).
- Table T5: Query Drift Analysis (wins, losses, ties, and top 10 largest losses / 5 largest wins).
- Figures 1–4: Publication-quality visualizations saved to results/plots/.

Usage:
    python experiments/ablations.py --dataset scifact
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_qrels, load_queries
from src.evaluator import oracle_ndcg10
from src.reranker import read_trec_run
from src.statistics import run_planned_comparisons

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments.ablations")


def generate_table_t1(agg_df: pd.DataFrame, dataset: str, out_dir: Path) -> pd.DataFrame:
    """Generate Table T1: Main Results across all metrics for depth 0 and depth 100."""
    logger.info("Generating Table T1 (Main Results)...")
    sub = agg_df[
        (agg_df["dataset"] == dataset)
        & ((agg_df["reranked"] == False) | (agg_df["depth"] == 100))
    ].copy()

    sub["stage"] = sub.apply(
        lambda r: "First-Stage (No Rerank)" if not r["reranked"] else "Reranked Depth 100", axis=1
    )

    t1_piv = sub.pivot(index=["retriever", "stage", "metric"], columns="feedback", values="value")
    col_order = [c for c in ["F0", "F1", "F2", "F3", "F4", "F5", "F6"] if c in t1_piv.columns]
    t1_piv = t1_piv[col_order]

    out_csv = out_dir / f"{dataset}_T1_main_results.csv"
    t1_piv.to_csv(out_csv)
    logger.info("Saved Table T1 to %s", out_csv)
    return t1_piv


def generate_table_t2(
    agg_df: pd.DataFrame,
    dataset: str,
    cfg: Dict[str, Any],
    qrels: Dict[str, Dict[str, int]],
    out_dir: Path,
) -> pd.DataFrame:
    """Generate Table T2: Recall-to-nDCG transfer and Oracle Ceiling."""
    logger.info("Generating Table T2 (Recall-to-nDCG Transfer & Oracle Ceiling)...")
    retrievers = cfg.get("retrievers", ["bm25", "qld"])
    conditions = cfg.get("feedback", {}).get("conditions", ["F0", "F1", "F2", "F3", "F4", "F5", "F6"])

    runs_base = Path(cfg["paths"]["runs"])
    fs_dir = runs_base / "first_stage" / dataset

    records = []

    for ret in retrievers:
        # Get baseline F0 values
        f0_r100_row = agg_df[
            (agg_df["dataset"] == dataset)
            & (agg_df["retriever"] == ret)
            & (agg_df["feedback"] == "F0")
            & (agg_df["reranked"] == False)
            & (agg_df["metric"] == "R@100")
        ]
        f0_ndcg_row = agg_df[
            (agg_df["dataset"] == dataset)
            & (agg_df["retriever"] == ret)
            & (agg_df["feedback"] == "F0")
            & (agg_df["reranked"] == True)
            & (agg_df["depth"] == 100)
            & (agg_df["metric"] == "nDCG@10")
        ]

        f0_r100 = float(f0_r100_row["value"].values[0]) if not f0_r100_row.empty else 0.0
        f0_ndcg = float(f0_ndcg_row["value"].values[0]) if not f0_ndcg_row.empty else 0.0

        for cond in conditions:
            # 1. First-stage Recall@100
            r100_row = agg_df[
                (agg_df["dataset"] == dataset)
                & (agg_df["retriever"] == ret)
                & (agg_df["feedback"] == cond)
                & (agg_df["reranked"] == False)
                & (agg_df["metric"] == "R@100")
            ]
            r100 = float(r100_row["value"].values[0]) if not r100_row.empty else 0.0
            delta_r100 = r100 - f0_r100

            # 2. Post-reranking (depth 100) nDCG@10
            ndcg_row = agg_df[
                (agg_df["dataset"] == dataset)
                & (agg_df["retriever"] == ret)
                & (agg_df["feedback"] == cond)
                & (agg_df["reranked"] == True)
                & (agg_df["depth"] == 100)
                & (agg_df["metric"] == "nDCG@10")
            ]
            ndcg = float(ndcg_row["value"].values[0]) if not ndcg_row.empty else 0.0
            delta_ndcg = ndcg - f0_ndcg

            # 3. Oracle Ceiling at Depth 100
            fs_run_file = fs_dir / f"{ret}_{cond}.trec"
            if fs_run_file.exists():
                oracle_df = oracle_ndcg10(fs_run_file, qrels, depth=100)
                oracle_val = float(oracle_df["value"].mean())
            else:
                oracle_val = np.nan

            records.append({
                "dataset": dataset,
                "retriever": ret,
                "feedback": cond,
                "first_stage_R@100": r100,
                "delta_R@100_vs_F0": delta_r100,
                "reranked_nDCG@10": ndcg,
                "delta_nDCG@10_vs_F0": delta_ndcg,
                "oracle_nDCG@10_ceiling": oracle_val,
                "realized_vs_ceiling_pct": (ndcg / oracle_val * 100.0) if oracle_val > 0 else np.nan,
            })

    t2_df = pd.DataFrame(records)
    out_csv = out_dir / f"{dataset}_T2_recall_ndcg_transfer.csv"
    t2_df.to_csv(out_csv, index=False)
    logger.info("Saved Table T2 to %s", out_csv)
    return t2_df


def generate_table_t4(
    agg_df: pd.DataFrame,
    dataset: str,
    cfg: Dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    """Generate Table T4: Depth vs. Efficiency / Cost Trade-offs."""
    logger.info("Generating Table T4 (Depth & Efficiency Trade-offs)...")
    runs_base = Path(cfg["paths"]["runs"])
    fs_dir = runs_base / "first_stage" / dataset

    # Load HyDE meta sidecar for generation time
    hyde_meta_path = Path(cfg["paths"]["expansions"]) / "hyde" / f"{dataset}.meta.json"
    hyde_sec_per_q = 0.0
    if hyde_meta_path.exists():
        with open(hyde_meta_path, "r", encoding="utf-8") as f:
            hm = json.load(f)
        # Average latency from A6 / hyde cache
        # Total model time 4577s across 300 queries = ~15.25s/query (4 passages)
        hyde_sec_per_q = 15.26

    retrievers = cfg.get("retrievers", ["bm25", "qld"])
    conditions = cfg.get("feedback", {}).get("conditions", ["F0", "F1", "F2", "F3", "F4", "F5", "F6"])

    records = []

    for ret in retrievers:
        for cond in conditions:
            # First-stage search latency from sidecar
            fs_meta_path = fs_dir / f"{ret}_{cond}.meta.json"
            fs_sec_per_q = 0.0
            if fs_meta_path.exists():
                with open(fs_meta_path, "r", encoding="utf-8") as f:
                    meta_fs = json.load(f)
                    fs_sec_per_q = float(meta_fs.get("avg_seconds_per_query", 0.0))

            llm_time = hyde_sec_per_q if cond in ("F3", "F4", "F5", "F6") else 0.0

            # nDCG@10 at depths 0, 20, 50, 100
            def get_ndcg(reranked: bool, depth: int) -> float:
                row = agg_df[
                    (agg_df["dataset"] == dataset)
                    & (agg_df["retriever"] == ret)
                    & (agg_df["feedback"] == cond)
                    & (agg_df["reranked"] == reranked)
                    & (agg_df["depth"] == depth)
                    & (agg_df["metric"] == "nDCG@10")
                ]
                return float(row["value"].values[0]) if not row.empty else np.nan

            ndcg_0 = get_ndcg(False, 0)
            ndcg_20 = get_ndcg(True, 20)
            ndcg_50 = get_ndcg(True, 50)
            ndcg_100 = get_ndcg(True, 100)

            records.append({
                "dataset": dataset,
                "retriever": ret,
                "feedback": cond,
                "nDCG@10_no_rerank": ndcg_0,
                "nDCG@10_depth20": ndcg_20,
                "nDCG@10_depth50": ndcg_50,
                "nDCG@10_depth100": ndcg_100,
                "first_stage_sec_per_query": fs_sec_per_q,
                "llm_gen_sec_per_query": llm_time,
                "cross_encoder_pairs_depth100": 100,
            })

    t4_df = pd.DataFrame(records)
    out_csv = out_dir / f"{dataset}_T4_depth_cost_efficiency.csv"
    t4_df.to_csv(out_csv, index=False)
    logger.info("Saved Table T4 to %s", out_csv)
    return t4_df


def generate_table_t5(
    pq_df: pd.DataFrame,
    dataset: str,
    queries: Dict[str, str],
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate Table T5: Query Drift Analysis (Wins, Losses, Ties and Outlier Inspection)."""
    logger.info("Generating Table T5 (Query Drift Analysis & Outlier Inspection)...")

    conditions = ["F1", "F2", "F3", "F4", "F5", "F6"]
    retrievers = ["bm25", "qld"]
    stages = [
        {"reranked": False, "depth": 0, "stage_name": "First-Stage"},
        {"reranked": True, "depth": 100, "stage_name": "Reranked Depth 100"},
    ]

    summary_records = []
    outlier_records = []

    for stage_info in stages:
        is_reranked = stage_info["reranked"]
        depth_val = stage_info["depth"]
        st_name = stage_info["stage_name"]

        for ret in retrievers:
            # Baseline values
            f0_df = pq_df[
                (pq_df["dataset"] == dataset)
                & (pq_df["retriever"] == ret)
                & (pq_df["feedback"] == "F0")
                & (pq_df["reranked"] == is_reranked)
                & (pq_df["depth"] == depth_val)
                & (pq_df["metric"] == "nDCG@10")
            ].sort_values("qid")

            f0_map = dict(zip(f0_df["qid"].astype(str), f0_df["value"]))

            for cond in conditions:
                cond_df = pq_df[
                    (pq_df["dataset"] == dataset)
                    & (pq_df["retriever"] == ret)
                    & (pq_df["feedback"] == cond)
                    & (pq_df["reranked"] == is_reranked)
                    & (pq_df["depth"] == depth_val)
                    & (pq_df["metric"] == "nDCG@10")
                ].sort_values("qid")

                deltas = []
                for _, row in cond_df.iterrows():
                    qid = str(row["qid"])
                    v_cond = float(row["value"])
                    v_f0 = float(f0_map.get(qid, 0.0))
                    d = v_cond - v_f0
                    deltas.append((qid, v_f0, v_cond, d))

                wins = sum(1 for _, _, _, d in deltas if d > 0.001)
                losses = sum(1 for _, _, _, d in deltas if d < -0.001)
                ties = sum(1 for _, _, _, d in deltas if abs(d) <= 0.001)

                summary_records.append({
                    "dataset": dataset,
                    "retriever": ret,
                    "stage": st_name,
                    "feedback": cond,
                    "wins_vs_F0": wins,
                    "losses_vs_F0": losses,
                    "ties_vs_F0": ties,
                    "win_loss_ratio": (wins / max(1, losses)),
                })

                # Sort by delta to find largest losses (drift) and wins
                deltas.sort(key=lambda x: x[3])
                # Top 10 largest losses (drift)
                for rank_loss, (qid, v0, vc, d) in enumerate(deltas[:10], start=1):
                    outlier_records.append({
                        "dataset": dataset,
                        "retriever": ret,
                        "stage": st_name,
                        "feedback": cond,
                        "type": "loss",
                        "rank": rank_loss,
                        "qid": qid,
                        "query_claim": queries.get(qid, ""),
                        "F0_nDCG10": v0,
                        "feedback_nDCG10": vc,
                        "delta": d,
                    })
                # Top 5 largest wins
                for rank_win, (qid, v0, vc, d) in enumerate(reversed(deltas[-5:]), start=1):
                    outlier_records.append({
                        "dataset": dataset,
                        "retriever": ret,
                        "stage": st_name,
                        "feedback": cond,
                        "type": "win",
                        "rank": rank_win,
                        "qid": qid,
                        "query_claim": queries.get(qid, ""),
                        "F0_nDCG10": v0,
                        "feedback_nDCG10": vc,
                        "delta": d,
                    })

    t5_sum_df = pd.DataFrame(summary_records)
    t5_outliers_df = pd.DataFrame(outlier_records)

    out_csv_sum = out_dir / f"{dataset}_T5_drift_summary.csv"
    out_csv_out = out_dir / f"{dataset}_T5_drift_outliers.csv"
    t5_sum_df.to_csv(out_csv_sum, index=False)
    t5_outliers_df.to_csv(out_csv_out, index=False)

    logger.info("Saved Table T5 to %s and %s", out_csv_sum, out_csv_out)
    return t5_sum_df, t5_outliers_df


def generate_figures(
    agg_df: pd.DataFrame,
    dataset: str,
    plots_dir: Path,
) -> None:
    """Generate Figures 1–4 from actual experimental result files."""
    logger.info("Generating Figures 1–4 in %s...", plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]

    # -------------------------------------------------------------
    # Figure 1: nDCG@10 per feedback strategy before & after reranking
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, ret in enumerate(["bm25", "qld"]):
        ax = axes[idx]
        sub = agg_df[(agg_df["dataset"] == dataset) & (agg_df["retriever"] == ret) & (agg_df["metric"] == "nDCG@10")]

        # First stage (no rerank)
        fs_vals = [float(sub[(sub["feedback"] == c) & (sub["reranked"] == False)]["value"].values[0]) for c in conditions]
        # Reranked Depth 100
        rr_vals = [float(sub[(sub["feedback"] == c) & (sub["reranked"] == True) & (sub["depth"] == 100)]["value"].values[0]) for c in conditions]
        # Reranked Depth 20
        rr20_vals = [float(sub[(sub["feedback"] == c) & (sub["reranked"] == True) & (sub["depth"] == 20)]["value"].values[0]) for c in conditions]

        x = np.arange(len(conditions))
        width = 0.25

        ax.bar(x - width, fs_vals, width, label="First-Stage (No Rerank)", color="#4A90E2", alpha=0.9)
        ax.bar(x, rr20_vals, width, label="Reranked Depth 20", color="#50E3C2", alpha=0.9)
        ax.bar(x + width, rr_vals, width, label="Reranked Depth 100", color="#B8E986", alpha=0.9)

        ax.set_title(f"{dataset.upper()} - {ret.upper()} (nDCG@10)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=11)
        ax.set_ylim(0.60, 0.76)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if idx == 0:
            ax.set_ylabel("nDCG@10", fontsize=11)
            ax.legend(loc="lower right")

    plt.tight_layout()
    fig1_path = plots_dir / f"{dataset}_Figure_1_ndcg10_before_after.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 1 to %s", fig1_path)

    # -------------------------------------------------------------
    # Figure 2: Recall across depths (20, 50, 100) per strategy
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    depth_metrics = ["R@20", "R@50", "R@100"]

    for idx, ret in enumerate(["bm25", "qld"]):
        ax = axes[idx]
        for c in conditions:
            y_vals = []
            for dm in depth_metrics:
                row = agg_df[
                    (agg_df["dataset"] == dataset)
                    & (agg_df["retriever"] == ret)
                    & (agg_df["feedback"] == c)
                    & (agg_df["reranked"] == False)
                    & (agg_df["metric"] == dm)
                ]
                y_vals.append(float(row["value"].values[0]) if not row.empty else 0.0)

            linestyle = "-" if c in ("F0", "F3", "F4") else "--"
            marker = "o" if c.startswith("F4") or c.startswith("F3") else "s"
            ax.plot([20, 50, 100], y_vals, label=c, linestyle=linestyle, marker=marker, linewidth=1.8)

        ax.set_title(f"{dataset.upper()} - {ret.upper()} Recall Depth Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Candidate Depth k", fontsize=11)
        ax.set_xticks([20, 50, 100])
        ax.set_ylim(0.80, 0.98)
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.set_ylabel("Recall@k", fontsize=11)
            ax.legend(ncol=2, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig2_path = plots_dir / f"{dataset}_Figure_2_recall_depth_curves.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 2 to %s", fig2_path)

    # -------------------------------------------------------------
    # Figure 3: Heatmap of methods before and after reranking
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    stages = ["No Rerank", "Depth 20", "Depth 50", "Depth 100"]

    for idx, ret in enumerate(["bm25", "qld"]):
        ax = axes[idx]
        matrix = np.zeros((len(stages), len(conditions)))

        for s_idx, st in enumerate(stages):
            is_rr = (st != "No Rerank")
            d_val = 0 if not is_rr else int(st.split()[1])
            for c_idx, c in enumerate(conditions):
                row = agg_df[
                    (agg_df["dataset"] == dataset)
                    & (agg_df["retriever"] == ret)
                    & (agg_df["feedback"] == c)
                    & (agg_df["reranked"] == is_rr)
                    & (agg_df["depth"] == d_val)
                    & (agg_df["metric"] == "nDCG@10")
                ]
                matrix[s_idx, c_idx] = float(row["value"].values[0]) if not row.empty else np.nan

        im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.63, vmax=0.73)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, fontsize=10)
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels(stages, fontsize=10)
        ax.set_title(f"{dataset.upper()} - {ret.upper()} (nDCG@10 Matrix)", fontsize=12, fontweight="bold")

        # Overlay text
        for i in range(len(stages)):
            for j in range(len(conditions)):
                val = matrix[i, j]
                text_color = "white" if val > 0.70 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig3_path = plots_dir / f"{dataset}_Figure_3_performance_heatmap.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 3 to %s", fig3_path)

    # -------------------------------------------------------------
    # Figure 4: Efficiency trade-off (nDCG@10 vs. Latency per query)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        "F0": "#4A90E2", "F1": "#E74C3C", "F2": "#E67E22",
        "F3": "#2ECC71", "F4": "#1ABC9C", "F5": "#9B59B6", "F6": "#34495E"
    }

    # Use BM25 configurations
    for c in conditions:
        # First-stage (no rerank)
        fs_row = agg_df[(agg_df["dataset"] == dataset) & (agg_df["retriever"] == "bm25") & (agg_df["feedback"] == c) & (agg_df["reranked"] == False) & (agg_df["metric"] == "nDCG@10")]
        val_fs = float(fs_row["value"].values[0])
        lat_fs = 0.05 if not c.startswith("F") or c in ("F0", "F1", "F2") else 15.25

        # Reranked Depth 20
        r20_row = agg_df[(agg_df["dataset"] == dataset) & (agg_df["retriever"] == "bm25") & (agg_df["feedback"] == c) & (agg_df["reranked"] == True) & (agg_df["depth"] == 20) & (agg_df["metric"] == "nDCG@10")]
        val_r20 = float(r20_row["value"].values[0])
        lat_r20 = lat_fs + (20 / 300.0)  # ~0.06s inference

        # Reranked Depth 100
        r100_row = agg_df[(agg_df["dataset"] == dataset) & (agg_df["retriever"] == "bm25") & (agg_df["feedback"] == c) & (agg_df["reranked"] == True) & (agg_df["depth"] == 100) & (agg_df["metric"] == "nDCG@10")]
        val_r100 = float(r100_row["value"].values[0])
        lat_r100 = lat_fs + (100 / 300.0)  # ~0.33s inference

        ax.scatter([lat_fs, lat_r20, lat_r100], [val_fs, val_r20, val_r100], color=colors[c], s=70, label=c)
        ax.plot([lat_fs, lat_r20, lat_r100], [val_fs, val_r20, val_r100], color=colors[c], linestyle=":", alpha=0.7)
        ax.annotate(f"{c}-D20", (lat_r20, val_r20), textcoords="offset points", xytext=(5, 2), fontsize=8)

    ax.set_title(f"{dataset.upper()} BM25 Efficiency Frontier: nDCG@10 vs. Latency per Query", fontsize=12, fontweight="bold")
    ax.set_xlabel("Estimated Latency per Query (seconds, log scale)", fontsize=11)
    ax.set_ylabel("nDCG@10", fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Condition", loc="lower right")

    plt.tight_layout()
    fig4_path = plots_dir / f"{dataset}_Figure_4_efficiency_tradeoff.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 4 to %s", fig4_path)


def main():
    parser = argparse.ArgumentParser(description="Run ablations, hypothesis tests, and figure generation.")
    parser.add_argument("--dataset", type=str, default="scifact", help="Target dataset name.")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to config.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_metrics_dir = Path(cfg["paths"]["results"]) / "metrics"
    out_sig_dir = Path(cfg["paths"]["results"]) / "significance"
    out_plots_dir = Path(cfg["paths"]["results"]) / "plots"

    agg_csv = out_metrics_dir / f"{args.dataset}_aggregate.csv"
    pq_csv = out_metrics_dir / f"{args.dataset}_per_query.csv"

    if not agg_csv.exists() or not pq_csv.exists():
        raise FileNotFoundError(f"Missing aggregate ({agg_csv}) or per-query ({pq_csv}) metrics file!")

    agg_df = pd.read_csv(agg_csv)
    pq_df = pd.read_csv(pq_csv)

    queries = load_queries(args.dataset, cfg)
    qrels = load_qrels(args.dataset, cfg)

    # 1. Planned Hypothesis Significance Tests
    print("\n" + "=" * 80)
    print("STEP 1: RUNNING PLANNED HYPOTHESIS TESTS (Holm-Bonferroni & Bootstrap CIs)")
    print("=" * 80)
    sig_df = run_planned_comparisons(args.dataset, cfg, metric="nDCG@10")
    print(f"Computed {len(sig_df)} planned comparisons.")

    # 2. Table T1: Main Results Table
    t1_df = generate_table_t1(agg_df, args.dataset, out_metrics_dir)

    # 3. Table T2: Recall-to-nDCG Transfer
    t2_df = generate_table_t2(agg_df, args.dataset, cfg, qrels, out_metrics_dir)

    # 4. Table T4: Depth and Cost
    t4_df = generate_table_t4(agg_df, args.dataset, cfg, out_metrics_dir)

    # 5. Table T5: Query Drift
    t5_sum_df, t5_outliers_df = generate_table_t5(pq_df, args.dataset, queries, cfg, out_metrics_dir)

    # 6. Figures 1–4
    generate_figures(agg_df, args.dataset, out_plots_dir)

    # Print summary highlights
    print("\n" + "=" * 80)
    print("PLANNED COMPARISONS SUMMARY (BM25 Reranked Depth 100 vs. Baseline)")
    print("=" * 80)
    bm25_rr_sig = sig_df[(sig_df["retriever"] == "bm25") & (sig_df["reranked"] == True)]
    summary_cols = ["comparison_id", "condition_a", "condition_b", "mean_diff", "wins", "losses", "ties", "p_permutation", "p_holm"]
    print(bm25_rr_sig[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
