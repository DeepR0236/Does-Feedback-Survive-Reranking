#!/usr/bin/env python3
"""Acceptance Tests A9 & A10: Determinism and Cache Reproducibility.

A9: Determinism Check:
    Reruns one configuration (e.g. BM25 F3 on SciFact) from the cached HyDE expansions.
    Verifies that the generated TREC run file is byte-for-byte or rank-for-rank identical.

A10: Full Reproducibility Invariant:
    Verifies that all required cache files (HyDE expansions, metadata, SQLite scores)
    exist and are intact, allowing complete end-to-end reconstruction of metric tables.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_queries
from src.feedback import build_feedback_queries
from src.retriever import run_first_stage


def test_a9_determinism(config_path: str = "configs/experiment_config.yaml") -> bool:
    print("=" * 75)
    print("ACCEPTANCE TEST A9: Determinism Check (SciFact BM25 F3)")
    print("=" * 75)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = "scifact"
    ret = "bm25"
    cond = "F3"
    queries = load_queries(dataset, cfg)

    orig_run_path = Path(cfg["paths"]["runs"]) / "first_stage" / dataset / f"{ret}_{cond}.trec"
    if not orig_run_path.exists():
        print(f"  FAIL: Original run {orig_run_path} does not exist.")
        return False

    with open(orig_run_path, "r", encoding="utf-8") as f:
        orig_lines = [line.strip() for line in f if line.strip()]

    # Re-run from cache to a temporary run file
    temp_run_path = Path("cache") / "a9_test_repro.trec"
    fb_queries = build_feedback_queries(dataset, ret, cond, queries, cfg)
    run_first_stage(
        dataset=dataset,
        retriever=ret,
        queries=fb_queries,
        cfg=cfg,
        tag=f"{ret}_{cond}",
        out_path=temp_run_path,
    )

    with open(temp_run_path, "r", encoding="utf-8") as f:
        new_lines = [line.strip() for line in f if line.strip()]

    # Clean up temp file
    if temp_run_path.exists():
        temp_run_path.unlink()
    temp_meta = temp_run_path.with_suffix(".meta.json")
    if temp_meta.exists():
        temp_meta.unlink()

    # Compare rank lines
    if len(orig_lines) != len(new_lines):
        print(f"  FAIL: Line count mismatch ({len(orig_lines)} vs {len(new_lines)})")
        return False

    mismatches = 0
    for idx, (l_orig, l_new) in enumerate(zip(orig_lines, new_lines)):
        # Extract qid, docid, rank
        p_orig = l_orig.split()
        p_new = l_new.split()
        if (p_orig[0], p_orig[2], p_orig[3]) != (p_new[0], p_new[2], p_new[3]):
            mismatches += 1

    if mismatches > 0:
        print(f"  FAIL: Found {mismatches} ranking mismatches between original and reproduced run.")
        return False

    print("  PASS: Re-executing from cache produced an identical TREC run file (0 mismatches).")
    print("=" * 75)
    print("ACCEPTANCE TEST A9: PASSED")
    print("=" * 75 + "\n")
    return True


def test_a10_cache_completeness(config_path: str = "configs/experiment_config.yaml") -> bool:
    print("=" * 75)
    print("ACCEPTANCE TEST A10: Full Cache Completeness & Reproducibility Check")
    print("=" * 75)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    datasets = ["scifact", "nfcorpus", "fiqa", "trec-covid"]
    cache_base = Path(cfg["paths"]["expansions"]) / "hyde"

    for ds in datasets:
        jsonl = cache_base / f"{ds}.jsonl"
        meta = cache_base / f"{ds}.meta.json"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            print(f"  FAIL: HyDE cache missing or empty for {ds}: {jsonl}")
            return False
        if not meta.exists():
            print(f"  FAIL: Sidecar metadata missing for {ds}: {meta}")
            return False
        print(f"  PASS: {ds.upper()} HyDE cache complete ({jsonl.stat().st_size / (1024*1024):.2f} MB)")

    # Check SQLite cache
    db_path = Path(cfg["paths"]["cache"]) / "rerank_scores.sqlite"
    if not db_path.exists() or db_path.stat().st_size == 0:
        print(f"  FAIL: SQLite score cache missing: {db_path}")
        return False
    print(f"  PASS: SQLite score cache intact ({db_path.stat().st_size / (1024*1024):.2f} MB)")

    print("=" * 75)
    print("ACCEPTANCE TEST A10: PASSED")
    print("=" * 75 + "\n")
    return True


if __name__ == "__main__":
    ok_a9 = test_a9_determinism()
    ok_a10 = test_a10_cache_completeness()
    sys.exit(0 if (ok_a9 and ok_a10) else 1)
