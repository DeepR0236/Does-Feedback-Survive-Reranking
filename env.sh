#!/usr/bin/env bash
# Environment configuration for "Does Feedback Survive Reranking?"
# Sets Java 21, redirects heavy caches to /data (493 GB free), and sets dummy OpenAI key for Pyserini 1.2.0.

# 1. Java 21 Home
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# 2. Heavy Storage Offloading to /data
export OLLAMA_MODELS=/data/ir_project_cache/ollama_models
export PYSERINI_CACHE=/data/ir_project_cache/pyserini
export HF_HOME=/data/ir_project_cache/huggingface

# 3. Pyserini 1.2.0 compatibility bypass (prevents missing key error on searcher import)
export OPENAI_API_KEY=dummy

# Ensure target cache directories exist on /data
mkdir -p "$OLLAMA_MODELS" 2>/dev/null || true
mkdir -p "$PYSERINI_CACHE" 2>/dev/null || true
mkdir -p "$HF_HOME" 2>/dev/null || true

echo "=== Project Environment Loaded ==="
echo "JAVA_HOME      = $JAVA_HOME"
echo "OLLAMA_MODELS  = $OLLAMA_MODELS"
echo "PYSERINI_CACHE = $PYSERINI_CACHE"
echo "HF_HOME        = $HF_HOME"
