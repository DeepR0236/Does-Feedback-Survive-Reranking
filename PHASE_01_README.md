# Phase 1: Environment & Project Foundation

## 1. Phase Objective
The objective of Phase 1 is to establish a verified, isolated, and reproducible software environment and project scaffolding for the research project **"Does Feedback Survive Reranking?"**.

By the end of Phase 1, you will have:
- OpenJDK 21 configured and verified (required by Apache Lucene 9.x via Pyserini).
- An isolated Python 3.10 virtual environment (`.venv`) with all required IR, neural, and scientific libraries.
- CUDA-enabled PyTorch recognizing your NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
- High-capacity storage configuration leveraging `/data` (488 GB available on NTFS) for heavy model weights and indexes, preserving your Ubuntu root partition (`/`).
- Permanent Ollama systemd service running under `hadoopuser` with `qwen2.5:3b-instruct` stored on `/data`.
- Repository directory structure and centralized configuration (`configs/experiment_config.yaml`).
- Initial data loader module (`src/data_loader.py`) for queries, qrels, and deterministic FiQA subsampling.
- Verification through **Acceptance Test A1** and data loader tests.

---

## 2. Storage Architecture: Dual-Drive Hybrid Layout

Your ASUS TUF laptop has:
* **Ubuntu Root (`/dev/nvme1n1p4`, ext4):** ~24 GB available. Used for source code, configuration files, git version control, virtual environment (`.venv`), and small output metadata.
* **Storage Partition (`/dev/nvme1n1p2`, NTFS mounted at `/data`):** ~488 GB available. Used for heavy caches and prebuilt indexes.

### Cache Offloading Mapping
We isolate all heavy external assets inside a single dedicated directory on `/data`:
```text
/data/ir_project_cache/
├── huggingface/          <- HuggingFace model weights (MiniLM Cross-Encoder, ~1 GB)
├── ir_datasets/          <- symlinked from ~/.ir_datasets
├── ollama_models/        <- Ollama models (Qwen2.5-3B, ~2.0 GB)
└── pyserini/             <- Prebuilt Lucene indexes (SciFact, NFCorpus, FiQA, TREC-COVID, ~2.5 GB)
```

This is controlled via standard environment variables defined in [env.sh](file:///home/hadoopuser/Desktop/InformationRetrieval/Project/Main/env.sh):
* `JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64`
* `OLLAMA_MODELS=/data/ir_project_cache/ollama_models`
* `PYSERINI_CACHE=/data/ir_project_cache/pyserini`
* `HF_HOME=/data/ir_project_cache/huggingface`
* `OPENAI_API_KEY=dummy` (Pyserini import compatibility bypass)

---

## 3. Ollama Systemd Configuration (NTFS Permissions)

Because `/data` is mounted with `uid=1001(hadoopuser), gid=1001(hadoopuser), dmask=0022`, only `hadoopuser` has write permissions on `/data/ir_project_cache/`. 

To ensure Ollama starts automatically on boot and has full permission to write model weights to `/data`, the service is installed at `/etc/systemd/system/ollama.service` configured with:
* `User=hadoopuser`
* `Group=hadoopuser`
* `Environment="OLLAMA_MODELS=/data/ir_project_cache/ollama_models"`
* `After=network-online.target data.mount`

---

## 4. Step-by-Step Execution Instructions

### Step 4.1: Configure Ollama Systemd Service (Best Practice)
1. In the terminal where `ollama serve` is currently running, stop it by pressing `Ctrl + C`.
2. Install and enable the systemd service:
```bash
sudo bash -c 'cat <<EOF > /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network-online.target data.mount
Wants=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=hadoopuser
Group=hadoopuser
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=/data/ir_project_cache/ollama_models"
Environment="OLLAMA_HOST=127.0.0.1:11434"

[Install]
WantedBy=multi-user.target
EOF'

# Reload systemd and start the background service
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sudo systemctl status ollama
```

### Step 4.2: Pull `qwen2.5:3b-instruct`
```bash
cd /home/hadoopuser/Desktop/InformationRetrieval/Project/Main
source env.sh
ollama pull qwen2.5:3b-instruct
```

### Step 4.3: Run Acceptance Test A1
```bash
cd /home/hadoopuser/Desktop/InformationRetrieval/Project/Main
source env.sh
source .venv/bin/activate

python tests/test_a1.py
```

### Step 4.4: Run Data Loader Verification
```bash
python tests/test_data_loader.py
```

---

## 5. Expected Test Outputs

### Acceptance Test A1 (`python tests/test_a1.py`)
```text
============================================================
 ACCEPTANCE TEST A1: Environment & Foundation Check
============================================================

[1/5] Checking Java version...
Java output:
openjdk version "21.0.13.1" 2026-11-17 ...
  --> PASS: Java 21 detected.

[2/5] Checking Pyserini import...
Pyserini version: 1.2.0
  --> PASS: Pyserini and Lucene modules imported successfully.

[3/5] Checking PyTorch and CUDA GPU availability...
PyTorch version: 2.6.0+cu124
CUDA Available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU (5.64 GB VRAM, 1 device(s))
  --> PASS: PyTorch detects CUDA GPU.

[4/5] Checking Ollama service and model response...
Ollama response: OK
  --> PASS: Ollama service is active and responsive.

[5/5] Checking sentence-transformers CrossEncoder...
Loading cross-encoder/ms-marco-MiniLM-L-6-v2...
CrossEncoder test pair score: -3.3325
  --> PASS: CrossEncoder loaded and scored test pair.

============================================================
 >>> ACCEPTANCE TEST A1 PASSED: All requirements satisfied! <<<
============================================================
```

### Data Loader Test (`python tests/test_data_loader.py`)
```text
============================================================
 Testing src/data_loader.py
============================================================

[1/4] Loading SciFact queries and qrels...
Loaded 300 queries and 300 qrels entries for SciFact.
  --> PASS: SciFact queries and qrels match expected size (300).

[2/4] Testing FiQA deterministic sampling...
  --> PASS: FiQA sample of 200 queries generated and saved to data/fiqa/sample_qids.txt.

[3/4] Initializing SciFact BM25 and QLD searchers...
  --> PASS: Both searchers initialized successfully.

[4/4] Extracting sample document text...
Sample docid: 31715818, text length: 1240 characters
  --> PASS: Document text extracted successfully.

============================================================
 >>> All data_loader tests passed! <<<
============================================================
```

---

## 6. Self-Assessment Checklist Before Phase 2
- [ ] Ollama service is running via systemd (`sudo systemctl status ollama` shows active).
- [ ] `ollama list` shows `qwen2.5:3b-instruct:latest`.
- [ ] Models are verified on `/data` (`ls /data/ir_project_cache/ollama_models/`).
- [ ] `python tests/test_a1.py` reports `ACCEPTANCE TEST A1 PASSED`.
- [ ] `python tests/test_data_loader.py` passes all 4 checks.
