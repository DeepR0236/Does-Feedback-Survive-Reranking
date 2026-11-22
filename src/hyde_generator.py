"""HyDE (Hypothetical Document Embeddings) passage generation module.

Interacts with the local Ollama server to generate synthetic passages for queries
using task-specific prompts, deterministic passage seeding, multi-threaded worker pools,
and append-only JSONL caching with QueryGym-style metadata tracking.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import zlib

import requests
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)

# Prompt templates from Appendix A of the specification
PROMPT_TEMPLATES: Dict[str, str] = {
    "scifact": (
        "Please write a scientific paper passage to support/refute the claim.\n"
        "Claim: {query}\n"
        "Passage:"
    ),
    "covid": (
        "Please write a scientific paper passage to answer the question.\n"
        "Question: {query}\n"
        "Passage:"
    ),
    "fiqa": (
        "Please write a financial article passage to answer the question.\n"
        "Question: {query}\n"
        "Passage:"
    ),
}


def build_prompt(dataset: str, query: str, cfg: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str]:
    """Construct prompt text, prompt_id, and prompt_sha1 for a dataset query.

    Args:
        dataset: Dataset identifier ('scifact', 'nfcorpus', 'fiqa', 'trec-covid').
        query: User query text.
        cfg: Optional configuration dictionary.

    Returns:
        Tuple of (formatted_prompt, prompt_id, prompt_sha1).
    """
    if cfg and "datasets" in cfg and dataset in cfg["datasets"]:
        prompt_id = cfg["datasets"][dataset].get("prompt", dataset)
    else:
        prompt_id = "covid" if dataset in ("nfcorpus", "trec-covid") else dataset

    template = PROMPT_TEMPLATES.get(prompt_id, PROMPT_TEMPLATES["covid"])
    prompt_text = template.format(query=query)
    prompt_sha1 = hashlib.sha1(template.encode("utf-8")).hexdigest()
    return prompt_text, prompt_id, prompt_sha1


def get_deterministic_seed(qid: str, passage_idx: int) -> int:
    """Compute deterministic seed for a passage: zlib.crc32(qid-idx) modulo 2**31."""
    key_str = f"{qid}-{passage_idx}"
    return zlib.crc32(key_str.encode("utf-8")) % (2**31)


def generate_passage(cfg: Dict[str, Any], prompt: str, seed: int) -> Dict[str, Any]:
    """Call Ollama /api/chat with deterministic seed, temperature, and num_predict.

    Implements the retry rule: if a passage is empty or under 20 characters,
    retries up to twice with a perturbed seed.

    Args:
        cfg: Configuration dictionary.
        prompt: Prompt string sent to the model.
        seed: Deterministic random seed integer.

    Returns:
        Dictionary with keys: text, seconds, tokens.
    """
    hyde_cfg = cfg.get("hyde", {})
    host = hyde_cfg.get("host", "http://localhost:11434")
    model = hyde_cfg.get("model", "qwen2.5:3b-instruct")
    temperature = float(hyde_cfg.get("temperature", 0.7))
    max_tokens = int(hyde_cfg.get("max_new_tokens", 160))
    url = f"{host.rstrip('/')}/api/chat"

    max_attempts = 3
    current_seed = seed

    for attempt in range(max_attempts):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": current_seed,
            },
        }

        t0 = time.perf_counter()
        res = requests.post(url, json=payload, timeout=120)
        t1 = time.perf_counter()

        if res.status_code != 200:
            raise RuntimeError(f"Ollama API error {res.status_code}: {res.text}")

        data = res.json()
        message = data.get("message", {})
        text = message.get("content", "").strip()
        tokens = int(data.get("eval_count", 0))
        elapsed = t1 - t0

        if len(text) >= 20:
            return {"text": text, "seconds": elapsed, "tokens": tokens, "seed": current_seed}

        logger.warning(
            "Generated passage too short (%d chars, attempt %d/%d). Retrying with new seed...",
            len(text),
            attempt + 1,
            max_attempts,
        )
        current_seed = (current_seed + 10007) % (2**31)

    # Return whatever was generated after retries
    return {"text": text, "seconds": elapsed, "tokens": tokens, "seed": current_seed}


def generate_all(dataset: str, cfg: Dict[str, Any], max_queries: Optional[int] = None) -> None:
    """Generate all HyDE passages for a dataset into an append-only JSONL cache.

    Resumable: reads existing (qid, passage_idx) pairs from disk and skips them.
    Thread-safe writes via immediate append and flush.

    Args:
        dataset: Target dataset ('scifact', 'nfcorpus', 'fiqa', 'trec-covid').
        cfg: Experiment configuration dictionary.
        max_queries: Optional limit for testing/speed checks.
    """
    from src.data_loader import load_queries

    queries = load_queries(dataset, cfg)
    if max_queries:
        queries = dict(list(queries.items())[:max_queries])

    num_passages = int(cfg.get("hyde", {}).get("num_passages", 4))
    num_workers = int(cfg.get("hyde", {}).get("workers", 2))
    cache_dir = Path(cfg["paths"]["expansions"]) / "hyde"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = cache_dir / f"{dataset}.jsonl"
    meta_path = cache_dir / f"{dataset}.meta.json"

    # Read existing cached entries to ensure idempotency & resumability
    existing_keys = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_keys.add((str(record["qid"]), int(record["passage_idx"])))
                    except Exception:
                        pass
        logger.info("Found %d already-cached passages in %s", len(existing_keys), jsonl_path)

    # Identify tasks to execute
    tasks = []
    for qid, qtext in queries.items():
        for p_idx in range(num_passages):
            if (qid, p_idx) not in existing_keys:
                seed = get_deterministic_seed(qid, p_idx)
                prompt_text, prompt_id, prompt_sha1 = build_prompt(dataset, qtext, cfg)
                tasks.append({
                    "qid": qid,
                    "qtext": qtext,
                    "passage_idx": p_idx,
                    "seed": seed,
                    "prompt_text": prompt_text,
                    "prompt_id": prompt_id,
                    "prompt_sha1": prompt_sha1,
                })

    if not tasks:
        logger.info("All %d passages for %s are already cached in %s", len(queries) * num_passages, dataset, jsonl_path)
        _write_meta_file(dataset, meta_path, cfg)
        return

    logger.info(
        "Generating %d remaining HyDE passages for %s (%d queries, %d passages/query, %d workers)...",
        len(tasks),
        dataset,
        len(queries),
        num_passages,
        num_workers,
    )

    import threading
    file_lock = threading.Lock()

    def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
        result = generate_passage(cfg, task["prompt_text"], task["seed"])
        record = {
            "dataset": dataset,
            "qid": task["qid"],
            "query": task["qtext"],
            "passage_idx": task["passage_idx"],
            "text": result["text"],
            "model": cfg.get("hyde", {}).get("model", "qwen2.5:3b-instruct"),
            "prompt_id": task["prompt_id"],
            "prompt_sha1": task["prompt_sha1"],
            "temperature": float(cfg.get("hyde", {}).get("temperature", 0.7)),
            "max_new_tokens": int(cfg.get("hyde", {}).get("max_new_tokens", 160)),
            "seed": result["seed"],
            "gen_seconds": result["seconds"],
            "output_tokens": result["tokens"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Thread-safe append and flush (Section 14: "Append and flush each result immediately")
        with file_lock:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()

        return record

    total_seconds = 0.0
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(tasks), desc=f"HyDE {dataset}", unit="passage"):
            rec = f.result()
            total_seconds += rec["gen_seconds"]
            total_tokens += rec["output_tokens"]

    avg_speed = total_tokens / max(1e-3, total_seconds)
    logger.info(
        "Generation complete for %s. Total tokens: %d, Throughput: %.1f tokens/sec",
        dataset,
        total_tokens,
        avg_speed,
    )

    _write_meta_file(dataset, meta_path, cfg)


def _write_meta_file(dataset: str, meta_path: Path, cfg: Dict[str, Any]) -> None:
    """Write sidecar metadata file documenting model, template, and configuration snapshot."""
    prompt_id = cfg.get("datasets", {}).get(dataset, {}).get("prompt", dataset)
    template = PROMPT_TEMPLATES.get(prompt_id, PROMPT_TEMPLATES["covid"])

    # Query Ollama version if available
    host = cfg.get("hyde", {}).get("host", "http://localhost:11434")
    ollama_ver = "unknown"
    try:
        r = requests.get(f"{host.rstrip('/')}/api/version", timeout=5)
        if r.status_code == 200:
            ollama_ver = r.json().get("version", "unknown")
    except Exception:
        pass

    meta_payload = {
        "dataset": dataset,
        "prompt_id": prompt_id,
        "prompt_template": template,
        "prompt_sha1": hashlib.sha1(template.encode("utf-8")).hexdigest(),
        "model": cfg.get("hyde", {}).get("model", "qwen2.5:3b-instruct"),
        "ollama_version": ollama_ver,
        "temperature": float(cfg.get("hyde", {}).get("temperature", 0.7)),
        "max_new_tokens": int(cfg.get("hyde", {}).get("max_new_tokens", 160)),
        "num_passages": int(cfg.get("hyde", {}).get("num_passages", 4)),
        "workers": int(cfg.get("hyde", {}).get("workers", 2)),
        "config_snapshot": cfg.get("hyde", {}),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    logger.info("Saved HyDE metadata snapshot to %s", meta_path)


def load_hyde(dataset: str, n: int, cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """Load cached HyDE passages for a dataset.

    Returns the first `n` passages per query grouped and sorted by passage_idx.

    Args:
        dataset: Dataset identifier.
        n: Number of passages to return per query.
        cfg: Configuration dictionary.

    Returns:
        Mapping of qid to list of passage strings.
    """
    cache_dir = Path(cfg["paths"]["expansions"]) / "hyde"
    jsonl_path = cache_dir / f"{dataset}.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"HyDE cache not found at {jsonl_path}. Run 'python -m src.hyde_generator --dataset {dataset}' first."
        )

    passages_by_qid: Dict[str, List[Tuple[int, str]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    qid = str(record["qid"])
                    idx = int(record["passage_idx"])
                    text = str(record["text"])
                    if qid not in passages_by_qid:
                        passages_by_qid[qid] = []
                    passages_by_qid[qid].append((idx, text))
                except Exception as e:
                    logger.warning("Error reading line in %s: %s", jsonl_path, e)

    # Sort by passage_idx and truncate to n
    result: Dict[str, List[str]] = {}
    for qid, items in passages_by_qid.items():
        items.sort(key=lambda x: x[0])
        result[qid] = [text for _, text in items[:n]]

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate HyDE passages via Ollama.")
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset name ('scifact', 'all', etc.).")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to YAML config.")
    parser.add_argument("--speed-test", action="store_true", help="Run Day 1 speed rule check on first 20 queries.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.speed_test:
        print("\n=== DAY 1 SPEED RULE TEST (First 20 Queries) ===")
        t0 = time.perf_counter()
        generate_all(args.dataset, cfg, max_queries=20)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        time_per_q = elapsed / 20.0
        total_queries = 873  # 300 + 323 + 200 + 50 across all 4 datasets
        projected_hours = (time_per_q * total_queries) / 3600.0
        print(f"\n20 queries elapsed: {elapsed:.2f}s ({time_per_q:.2f}s/query)")
        print(f"Projected time for all 873 queries (N=4): {projected_hours:.2f} hours")
        if projected_hours > 6.0:
            print("  --> SPEED RULE TRIGGERED: Projected time > 6 hours. Switch to N=2.")
        else:
            print("  --> SPEED RULE PASS: Projected time <= 6 hours. Proceed with N=4.")
        return

    if args.dataset == "all":
        datasets = list(cfg["datasets"].keys())
    else:
        datasets = [args.dataset]

    for ds in datasets:
        generate_all(ds, cfg)


if __name__ == "__main__":
    main()
