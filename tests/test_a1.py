#!/usr/bin/env python3
"""Acceptance Test A1: Environment & Foundation Verification.

Verifies:
1. Java 21 is available and active.
2. Pyserini 1.2.0 imports cleanly and interacts with PyJNIus.
3. PyTorch detects the NVIDIA GPU (CUDA available).
4. Ollama server is responding and generates a test response.
5. sentence-transformers CrossEncoder loads 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
"""

import os
import subprocess
import sys
import requests

# Set defaults if not provided in environment
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
os.environ.setdefault("OPENAI_API_KEY", "dummy")


def check_java() -> bool:
    print("\n[1/5] Checking Java version...")
    java_bin = "java"
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        cand = os.path.join(java_home, "bin", "java")
        if os.path.exists(cand):
            java_bin = cand
    try:
        proc = subprocess.run([java_bin, "-version"], capture_output=True, text=True, check=True)
        version_output = proc.stderr or proc.stdout
        print(f"Java output:\n{version_output.strip()}")
        if "21." in version_output or '"21' in version_output:
            print("  --> PASS: Java 21 detected.")
            return True
        else:
            print(f"  --> FAIL: Java 21 required, but found different version. Check JAVA_HOME.")
            return False
    except Exception as e:
        print(f"  --> FAIL: Could not run java: {e}")
        return False


def check_pyserini() -> bool:
    print("\n[2/5] Checking Pyserini import...")
    try:
        import pyserini
        import importlib.metadata
        from pyserini.search.lucene import LuceneSearcher
        from pyserini.index.lucene import LuceneIndexReader
        ver = importlib.metadata.version("pyserini")
        print(f"Pyserini version: {ver}")
        print("  --> PASS: Pyserini and Lucene modules imported successfully.")
        return True
    except Exception as e:
        print(f"  --> FAIL: Failed to import Pyserini: {e}")
        return False


def check_torch_gpu() -> bool:
    print("\n[3/5] Checking PyTorch and CUDA GPU availability...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU: {device_name} ({vram_gb:.2f} GB VRAM, {device_count} device(s))")
            print("  --> PASS: PyTorch detects CUDA GPU.")
            return True
        else:
            print("  --> FAIL: PyTorch cannot detect GPU (CUDA not available).")
            return False
    except Exception as e:
        print(f"  --> FAIL: PyTorch check failed: {e}")
        return False


def check_ollama() -> bool:
    print("\n[4/5] Checking Ollama service and model response...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:3b-instruct",
        "prompt": "Reply with only the word 'OK'.",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 10}
    }
    try:
        res = requests.post(url, json=payload, timeout=90)
        if res.status_code == 200:
            data = res.json()
            response_text = data.get("response", "").strip()
            print(f"Ollama response: {response_text}")
            print("  --> PASS: Ollama service is active and responsive.")
            return True
        else:
            print(f"  --> FAIL: Ollama returned HTTP status {res.status_code}: {res.text}")
            return False
    except Exception as e:
        print(f"  --> FAIL: Failed to communicate with Ollama at localhost:11434: {e}")
        print("      Make sure Ollama is installed and running (`ollama serve`).")
        return False


def check_cross_encoder() -> bool:
    print("\n[5/5] Checking sentence-transformers CrossEncoder...")
    try:
        from sentence_transformers import CrossEncoder
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"Loading {model_name}...")
        model = CrossEncoder(model_name)
        score = model.predict([("test query", "test document text")])
        print(f"CrossEncoder test pair score: {score[0]:.4f}")
        print("  --> PASS: CrossEncoder loaded and scored test pair.")
        return True
    except Exception as e:
        print(f"  --> FAIL: Failed to load CrossEncoder: {e}")
        return False


def main():
    print("=" * 60)
    print(" ACCEPTANCE TEST A1: Environment & Foundation Check")
    print("=" * 60)
    results = [
        check_java(),
        check_pyserini(),
        check_torch_gpu(),
        check_ollama(),
        check_cross_encoder(),
    ]
    print("\n" + "=" * 60)
    if all(results):
        print(" >>> ACCEPTANCE TEST A1 PASSED: All requirements satisfied! <<<")
        print("=" * 60)
        sys.exit(0)
    else:
        failed_count = results.count(False)
        print(f" >>> ACCEPTANCE TEST A1 FAILED: {failed_count}/5 checks failed. <<<")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
