#!/usr/bin/env python3
"""Validation test for src/data_loader.py.

Verifies:
1. Loading SciFact queries and qrels.
2. FiQA deterministic 200-query sampling (seed 42).
3. LuceneSearcher initialization for BM25 and QLD.
4. Raw document extraction.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure environment defaults
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

import yaml

from src.data_loader import (
    load_queries,
    load_qrels,
    fiqa_sample_qids,
    get_searcher,
    get_doc_text,
)


def main():
    print("=" * 60)
    print(" Testing src/data_loader.py")
    print("=" * 60)

    cfg_path = Path("configs/experiment_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Test SciFact queries and qrels
    print("\n[1/4] Loading SciFact queries and qrels...")
    queries = load_queries("scifact", cfg)
    qrels = load_qrels("scifact", cfg)
    print(f"Loaded {len(queries)} queries and {len(qrels)} qrels entries for SciFact.")
    assert len(queries) == 300, f"Expected 300 queries for SciFact, got {len(queries)}"
    assert len(qrels) == 300, f"Expected 300 qrels for SciFact, got {len(qrels)}"
    print("  --> PASS: SciFact queries and qrels match expected size (300).")

    # 2. Test FiQA sample
    print("\n[2/4] Testing FiQA deterministic sampling...")
    sample_qids = fiqa_sample_qids(cfg)
    assert len(sample_qids) == 200, f"Expected 200 FiQA sampled queries, got {len(sample_qids)}"
    sample_file = Path("data/fiqa/sample_qids.txt")
    assert sample_file.exists(), f"{sample_file} should exist after sampling"
    print(f"  --> PASS: FiQA sample of {len(sample_qids)} queries generated and saved to {sample_file}.")

    # 3. Test LuceneSearcher initialization
    print("\n[3/4] Initializing SciFact BM25 and QLD searchers...")
    searcher_bm25 = get_searcher("scifact", "bm25", cfg)
    searcher_qld = get_searcher("scifact", "qld", cfg)
    print("  --> PASS: Both searchers initialized successfully.")

    # 4. Test doc extraction
    print("\n[4/4] Extracting sample document text...")
    hits = searcher_bm25.search("hypertension blood pressure", k=1)
    assert len(hits) > 0, "BM25 search returned 0 hits on SciFact"
    docid = hits[0].docid
    text = get_doc_text(searcher_bm25, docid)
    print(f"Sample docid: {docid}, text length: {len(text)} characters")
    assert len(text) > 0, "Document text is empty"
    print(f"Snippet: {text[:120]}...")
    print("  --> PASS: Document text extracted successfully.")

    print("\n" + "=" * 60)
    print(" >>> All data_loader tests passed! <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()
