#!/usr/bin/env python3
"""Acceptance Test A6: HyDE Passage Generation & Cache Verification.

Verifies:
1. Cache completeness: all (qid, passage_idx) pairs present (300 queries * 4 passages = 1200 records for SciFact).
2. Non-emptiness: no empty passages; every passage has >= 20 characters.
3. Metadata integrity: token counts and generation durations logged for every record.
4. Sidecar metadata file (scifact.meta.json) exists with valid configuration snapshot and prompt SHA1.
5. Eye check: prints 5 representative generated passages alongside query claims for visual verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_queries


def test_a6(dataset: str = "scifact", config_path: str = "configs/experiment_config.yaml") -> bool:
    print("=" * 75)
    print(f"ACCEPTANCE TEST A6: HyDE Passage Cache Verification ({dataset.upper()})")
    print("=" * 75)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cache_dir = Path(cfg["paths"]["expansions"]) / "hyde"
    jsonl_path = cache_dir / f"{dataset}.jsonl"
    meta_path = cache_dir / f"{dataset}.meta.json"

    # Check 1: Files exist
    print(f"[Check 1] Checking file existence...")
    if not jsonl_path.exists():
        print(f"  FAIL: JSONL cache not found at {jsonl_path}")
        return False
    print(f"  PASS: JSONL cache exists: {jsonl_path}")

    if not meta_path.exists():
        print(f"  FAIL: Sidecar metadata file not found at {meta_path}")
        return False
    print(f"  PASS: Metadata file exists: {meta_path}")

    # Check 2: Metadata file contents
    print(f"\n[Check 2] Validating sidecar metadata...")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    required_meta_keys = ["dataset", "prompt_id", "prompt_sha1", "model", "temperature", "max_new_tokens", "num_passages"]
    for k in required_meta_keys:
        if k not in meta_data:
            print(f"  FAIL: Missing key '{k}' in metadata sidecar")
            return False
    print(f"  PASS: Model: {meta_data['model']}, Prompt SHA1: {meta_data['prompt_sha1'][:10]}..., Temp: {meta_data['temperature']}")

    # Check 3: Load expected queries
    queries = load_queries(dataset, cfg)
    num_queries = len(queries)
    num_passages = int(cfg.get("hyde", {}).get("num_passages", 4))
    expected_total = num_queries * num_passages
    print(f"\n[Check 3] Dataset has {num_queries} queries; expecting {expected_total} passages ({num_passages}/query)...")

    # Read all records from JSONL
    records = []
    seen_keys = set()
    short_or_empty = []
    missing_fields = []
    tokens_logged = True

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  FAIL: JSON decode error at line {line_num}: {e}")
                return False

            qid = str(rec.get("qid"))
            pidx = rec.get("passage_idx")
            text = rec.get("text", "")
            tokens = rec.get("output_tokens")
            gen_sec = rec.get("gen_seconds")

            if qid not in queries:
                print(f"  WARNING: Unknown qid '{qid}' found at line {line_num}")

            key = (qid, pidx)
            if key in seen_keys:
                print(f"  FAIL: Duplicate passage key: {key} at line {line_num}")
                return False
            seen_keys.add(key)

            if not text or len(text.strip()) < 20:
                short_or_empty.append((key, len(text.strip()) if text else 0))

            if tokens is None or gen_sec is None:
                missing_fields.append(key)

            records.append(rec)

    print(f"  Total records read: {len(records)} (Unique keys: {len(seen_keys)})")

    # Check 4: Completeness
    print(f"\n[Check 4] Checking coverage of all (qid, passage_idx) pairs...")
    missing_keys = []
    for qid in queries:
        for pidx in range(num_passages):
            if (qid, pidx) not in seen_keys:
                missing_keys.append((qid, pidx))

    if missing_keys:
        print(f"  FAIL: Missing {len(missing_keys)} expected passage keys! Examples: {missing_keys[:5]}")
        return False
    print(f"  PASS: 100% complete! All {expected_total} (qid, passage_idx) pairs are present.")

    # Check 5: Non-emptiness and logging
    print(f"\n[Check 5] Validating passage length and metrics logging...")
    if short_or_empty:
        print(f"  FAIL: Found {len(short_or_empty)} passages with < 20 characters! Examples: {short_or_empty[:5]}")
        return False
    print(f"  PASS: All {len(records)} passages exceed the 20-character minimum.")

    if missing_fields:
        print(f"  FAIL: Found {len(missing_fields)} records missing token or latency logging! Examples: {missing_fields[:5]}")
        return False
    print(f"  PASS: Token counts and generation latencies are properly logged.")

    # Summary statistics
    all_tokens = [r["output_tokens"] for r in records]
    all_seconds = [r["gen_seconds"] for r in records]
    avg_tokens = sum(all_tokens) / len(all_tokens)
    avg_seconds = sum(all_seconds) / len(all_seconds)
    total_tokens = sum(all_tokens)
    total_time = sum(all_seconds)

    print(f"\n--- HyDE Generation Statistics ---")
    print(f"  Total passages:        {len(records)}")
    print(f"  Total tokens:          {total_tokens:,}")
    print(f"  Total model time:      {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"  Mean tokens/passage:   {avg_tokens:.1f}")
    print(f"  Mean latency/passage:  {avg_seconds:.2f}s")
    print(f"  Mean throughput:       {total_tokens / total_time:.1f} tokens/sec")

    # Check 6: Visual Inspection (Eye Check on 5 passages)
    print(f"\n[Check 6] Visual inspection of 5 sample passages for sense:")
    sample_qids = list(queries.keys())[:5]
    for i, qid in enumerate(sample_qids, 1):
        sample_rec = next(r for r in records if r["qid"] == qid and r["passage_idx"] == 0)
        print(f"\n--- Sample {i} (QID: {qid}, Passage 0) ---")
        print(f"Claim/Query:  {sample_rec['query']}")
        print(f"Seed:         {sample_rec['seed']}")
        print(f"Tokens/Time:  {sample_rec['output_tokens']} tokens in {sample_rec['gen_seconds']:.2f}s")
        print(f"Passage Text:\n{sample_rec['text']}")

    print("\n" + "=" * 75)
    print("ACCEPTANCE TEST A6: PASSED")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = test_a6()
    sys.exit(0 if success else 1)
