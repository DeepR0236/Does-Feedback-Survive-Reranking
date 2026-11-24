"""Statistical Significance & Hypothesis Testing Module.

Implements paired permutation tests, Wilcoxon signed-rank tests, bootstrap
confidence intervals, Holm-Bonferroni correction, win/loss/tie counting,
and execution of planned comparisons as specified in Section 7 & 14.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation test on the mean difference (Section 14).

    Randomly flips the signs of paired differences d_i = a_i - b_i.
    Returns (count + 1) / (n_perm + 1).

    Args:
        a: Array-like of per-query metric values for condition A.
        b: Array-like of per-query metric values for condition B.
        n_perm: Number of Monte Carlo sign permutations (default 10,000).
        seed: Random seed for exact determinism (default 42).

    Returns:
        Two-sided p-value float in range (0, 1].
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    if len(a_arr) != len(b_arr):
        raise ValueError(f"Arrays must have identical length, got {len(a_arr)} and {len(b_arr)}")

    diff = a_arr - b_arr
    n = len(diff)
    if n == 0:
        return 1.0

    observed_t = abs(float(np.mean(diff)))
    if observed_t < 1e-12:
        return 1.0

    rng = np.random.default_rng(seed)
    # Generate random +1 / -1 signs of shape (n_perm, n)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.abs(np.mean(signs * diff, axis=1))

    # Two-sided count: permuted mean >= observed mean (with floating point tolerance)
    count = int(np.sum(perm_means >= (observed_t - 1e-12)))

    return float((count + 1) / (n_perm + 1))


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank test.

    Returns p-value. If all differences are zero, returns 1.0.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    diff = a_arr - b_arr
    if np.all(np.abs(diff) < 1e-12):
        return 1.0

    try:
        res = stats.wilcoxon(a_arr, b_arr, zero_method="pratt", alternative="two-sided")
        return float(res.pvalue)
    except Exception as e:
        logger.warning("Wilcoxon test failed: %s. Returning 1.0.", e)
        return 1.0


def bootstrap_ci(
    diff: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval for the mean difference.

    Args:
        diff: Array-like of paired differences (a - b).
        n_boot: Number of bootstrap resamples (default 10,000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed (default 42).

    Returns:
        Tuple of (ci_low, ci_high).
    """
    diff_arr = np.asarray(diff, dtype=np.float64)
    n = len(diff_arr)
    if n == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diff_arr[boot_indices], axis=1)

    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_low = float(np.percentile(boot_means, low_pct))
    ci_high = float(np.percentile(boot_means, high_pct))

    return (ci_low, ci_high)


def win_loss_tie(
    a: Sequence[float],
    b: Sequence[float],
    tol: float = 0.001,
) -> Tuple[int, int, int]:
    """Count queries where condition A wins, loses, or ties against condition B.

    Args:
        a: Array of per-query metric values for condition A.
        b: Array of per-query metric values for condition B.
        tol: Practical equivalence threshold (default 0.001).

    Returns:
        Tuple of (wins, losses, ties).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    diff = a_arr - b_arr
    wins = int(np.sum(diff > tol))
    losses = int(np.sum(diff < -tol))
    ties = int(np.sum(np.abs(diff) <= tol))

    return (wins, losses, ties)


def holm(pvals: Sequence[float]) -> List[float]:
    """Step-down Holm-Bonferroni family-wise error rate correction.

    Enforces monotonicity: adjusted p_val is non-decreasing in rank order.

    Args:
        pvals: Sequence of raw p-values.

    Returns:
        List of adjusted p-values in the original input order.
    """
    p_arr = np.asarray(pvals, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return []

    # Sort ascending
    sort_idx = np.argsort(p_arr)
    sorted_p = p_arr[sort_idx]

    # Step-down multipliers: (m, m-1, m-2, ..., 1)
    multipliers = np.arange(m, 0, -1, dtype=np.float64)
    adjusted_sorted = np.empty(m, dtype=np.float64)

    running_max = 0.0
    for i in range(m):
        raw_adj = sorted_p[i] * multipliers[i]
        running_max = max(running_max, raw_adj)
        adjusted_sorted[i] = min(1.0, running_max)

    # Restore original ordering
    adjusted_p = np.empty(m, dtype=np.float64)
    adjusted_p[sort_idx] = adjusted_sorted

    return adjusted_p.tolist()


# List of 11 planned comparisons from Section 7
PLANNED_COMPARISONS: List[Tuple[str, str, str, str]] = [
    ("CMP01", "F1", "F0", "Rocchio vs. Baseline"),
    ("CMP02", "F2", "F0", "RM3 vs. Baseline"),
    ("CMP03", "F3", "F0", "HyDE vs. Baseline"),
    ("CMP04", "F4", "F0", "HyDE+Rocchio vs. Baseline"),
    ("CMP05", "F5", "F0", "HyDE+RM3 vs. Baseline"),
    ("CMP06", "F6", "F0", "Naive Concat vs. Baseline"),
    ("CMP07", "F4", "F1", "HyDE+Rocchio vs. Real Rocchio"),
    ("CMP08", "F5", "F2", "HyDE+RM3 vs. Real RM3"),
    ("CMP09", "F4", "F3", "HyDE+Rocchio vs. HyDE Centroid"),
    ("CMP10", "F5", "F3", "HyDE+RM3 vs. HyDE Centroid"),
    ("CMP11", "F6", "F4", "Naive Concat vs. HyDE+Rocchio"),
]


def run_planned_comparisons(
    dataset: str,
    cfg: Dict[str, Any],
    metric: str = "nDCG@10",
    n_perm: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all planned hypothesis tests and save to results/significance/{dataset}_planned.csv.

    Follows Section 7 & 14:
    - Planned comparisons: F0 vs F1-F6; F1 vs F4; F2 vs F5; F3 vs F4 & F5; F6 vs F4.
    - Evaluated before reranking (depth 0) and after reranking (depth 100).
    - Permutation test (10,000 perms), Wilcoxon signed-rank test.
    - 95% bootstrap confidence intervals.
    - Step-down Holm correction applied across the planned list.

    Args:
        dataset: Target dataset name.
        cfg: Experiment configuration dictionary.
        metric: Target evaluation metric (default 'nDCG@10').
        n_perm: Number of permutations.
        seed: Random seed.

    Returns:
        DataFrame of statistical comparison results.
    """
    per_query_path = Path(cfg["paths"]["results"]) / "metrics" / f"{dataset}_per_query.csv"
    if not per_query_path.exists():
        raise FileNotFoundError(f"Per-query metrics not found: {per_query_path}")

    pq_df = pd.read_csv(per_query_path)
    sig_dir = Path(cfg["paths"]["results"]) / "significance"
    sig_dir.mkdir(parents=True, exist_ok=True)
    out_csv = sig_dir / f"{dataset}_planned.csv"

    retrievers = cfg.get("retrievers", ["bm25", "qld"])
    stages = [
        {"reranked": False, "depth": 0, "stage_name": "First-Stage (No Rerank)"},
        {"reranked": True, "depth": 100, "stage_name": "Reranked Depth 100"},
    ]

    records: List[Dict[str, Any]] = []

    for stage_info in stages:
        is_reranked = stage_info["reranked"]
        depth_val = stage_info["depth"]

        for ret in retrievers:
            stage_records: List[Dict[str, Any]] = []
            raw_pvals: List[float] = []

            for cmp_id, cond_a, cond_b, desc in PLANNED_COMPARISONS:
                # Query values for cond_a
                sub_a = pq_df[
                    (pq_df["dataset"] == dataset)
                    & (pq_df["retriever"] == ret)
                    & (pq_df["feedback"] == cond_a)
                    & (pq_df["reranked"] == is_reranked)
                    & (pq_df["depth"] == depth_val)
                    & (pq_df["metric"] == metric)
                ].sort_values("qid")

                # Query values for cond_b
                sub_b = pq_df[
                    (pq_df["dataset"] == dataset)
                    & (pq_df["retriever"] == ret)
                    & (pq_df["feedback"] == cond_b)
                    & (pq_df["reranked"] == is_reranked)
                    & (pq_df["depth"] == depth_val)
                    & (pq_df["metric"] == metric)
                ].sort_values("qid")

                if sub_a.empty or sub_b.empty:
                    logger.warning("Missing data for comparison %s (%s vs %s)", cmp_id, cond_a, cond_b)
                    continue

                # Align by qid
                merged = pd.merge(sub_a[["qid", "value"]], sub_b[["qid", "value"]], on="qid", suffixes=("_a", "_b"))
                vals_a = merged["value_a"].values
                vals_b = merged["value_b"].values
                diff = vals_a - vals_b

                mean_diff = float(np.mean(diff))
                median_diff = float(np.median(diff))
                wins, losses, ties = win_loss_tie(vals_a, vals_b, tol=0.001)

                p_perm = paired_permutation_test(vals_a, vals_b, n_perm=n_perm, seed=seed)
                p_wilc = wilcoxon_test(vals_a, vals_b)
                ci_low, ci_high = bootstrap_ci(diff, n_boot=n_perm, alpha=0.05, seed=seed)

                raw_pvals.append(p_perm)

                stage_records.append({
                    "comparison_id": cmp_id,
                    "dataset": dataset,
                    "retriever": ret,
                    "reranked": is_reranked,
                    "depth": depth_val,
                    "metric": metric,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "description": desc,
                    "mean_diff": mean_diff,
                    "median_diff": median_diff,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "p_permutation": p_perm,
                    "p_wilcoxon": p_wilc,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                })

            # Apply Holm correction across this planned list
            holm_pvals = holm(raw_pvals)
            for rec, p_h in zip(stage_records, holm_pvals):
                rec["p_holm"] = p_h
                records.append(rec)

    cols = [
        "comparison_id", "dataset", "retriever", "reranked", "depth", "metric",
        "condition_a", "condition_b", "description",
        "mean_diff", "median_diff", "wins", "losses", "ties",
        "p_permutation", "p_wilcoxon", "p_holm", "ci_low", "ci_high"
    ]
    res_df = pd.DataFrame(records)[cols]
    res_df.to_csv(out_csv, index=False)
    logger.info("Saved %d planned comparison results to %s", len(res_df), out_csv)

    return res_df
