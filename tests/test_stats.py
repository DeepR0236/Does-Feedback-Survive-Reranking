#!/usr/bin/env python3
"""Unit and Acceptance Tests for Statistical Methods.

Validates:
1. Paired permutation test: returns 1.0 for identical data, < 0.01 for clearly shifted distributions.
2. Bootstrap confidence interval: correctly bounds the mean difference.
3. Win/Loss/Tie: partitions sample size exactly (wins + losses + ties == N).
4. Holm correction: non-decreasing monotonicity and correct step-down bounding.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistics import bootstrap_ci, holm, paired_permutation_test, wilcoxon_test, win_loss_tie


def test_statistics() -> bool:
    print("=" * 75)
    print("TESTING STATISTICAL METHODS (src/statistics.py)")
    print("=" * 75)

    rng = np.random.default_rng(42)

    # 1. Permutation test on identical data
    print("[Check 1] Permutation test on identical data...")
    a = [0.5, 0.6, 0.7, 0.8]
    p_ident = paired_permutation_test(a, a, n_perm=1000)
    print(f"  Identical data p-value: {p_ident:.4f}")
    assert p_ident == 1.0, f"Expected 1.0, got {p_ident}"
    print("  PASS: Identical data returns p=1.0")

    # 2. Permutation test on clearly different data
    print("\n[Check 2] Permutation test on clearly different data...")
    x = rng.normal(loc=0.8, scale=0.05, size=50)
    y = rng.normal(loc=0.4, scale=0.05, size=50)
    p_diff = paired_permutation_test(x, y, n_perm=5000, seed=42)
    print(f"  Separated data p-value: {p_diff:.6f}")
    assert p_diff < 0.01, f"Expected p < 0.01, got {p_diff}"
    print("  PASS: Separated data rejected with p < 0.01")

    # 3. Bootstrap CI
    print("\n[Check 3] Bootstrap confidence interval...")
    diff = x - y
    true_mean = np.mean(diff)
    ci_low, ci_high = bootstrap_ci(diff, n_boot=2000, alpha=0.05, seed=42)
    print(f"  True mean: {true_mean:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    assert ci_low <= true_mean <= ci_high, f"True mean {true_mean} not in CI [{ci_low}, {ci_high}]"
    print("  PASS: Bootstrap CI successfully encloses true mean difference.")

    # 4. Win / Loss / Tie
    print("\n[Check 4] Win/Loss/Tie partition...")
    wins, losses, ties = win_loss_tie(x, y, tol=0.001)
    print(f"  Wins: {wins}, Losses: {losses}, Ties: {ties} (Total: {wins+losses+ties})")
    assert wins + losses + ties == len(x), "Win/loss/tie does not partition sample!"
    assert wins == len(x), f"Expected 50 wins, got {wins}"
    print("  PASS: Win/loss/tie properly partitions queries.")

    # 5. Holm-Bonferroni correction
    print("\n[Check 5] Holm-Bonferroni step-down correction...")
    raw_p = [0.001, 0.012, 0.040, 0.150]
    adj_p = holm(raw_p)
    print(f"  Raw p-values:  {raw_p}")
    print(f"  Holm p-values: {adj_p}")
    # Monotonicity check
    for i in range(len(adj_p) - 1):
        assert adj_p[i] <= adj_p[i+1], f"Monotonicity violated: {adj_p[i]} > {adj_p[i+1]}"
        assert raw_p[i] <= adj_p[i], f"Adjusted p is smaller than raw p: {adj_p[i]} < {raw_p[i]}"
    print("  PASS: Holm correction satisfies monotonicity and bounding constraints.")

    print("\n" + "=" * 75)
    print("ALL STATISTICAL UNIT TESTS PASSED!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    ok = test_statistics()
    sys.exit(0 if ok else 1)
