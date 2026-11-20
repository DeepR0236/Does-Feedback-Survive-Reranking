# Phase 2: Retrieval & Evaluation Foundation

## 1. Phase Objective
The objective of Phase 2 is to establish the baseline first-stage retrieval mechanisms and evaluation infrastructure for the research project **"Does Feedback Survive Reranking?"**.

By the end of Phase 2, you will have:
- Implemented first-stage retrieval (`src/retriever.py`) supporting both **BM25** ($k_1=0.9, b=0.4$) and **Query Likelihood with Dirichlet smoothing** ($\mu=1000$).
- Produced standard 6-column TREC run files saving the top 1000 candidates per query.
- Implemented execution metadata logging (`.meta.json`) capturing exact hyperparameters, package versions, and per-query search latencies.
- Implemented a rigorous evaluation engine (`src/evaluator.py`) using `ir_measures` to compute all 12 target metrics (including $\text{nDCG}@10$, $\text{Recall}@100$, and $\text{Judged}@10$).
- Implemented the baseline experiment runner (`experiments/baseline.py`) for the unexpanded control condition ($F_0$).
- Verified **Acceptance Test A2** (BM25 SciFact baseline sanity) and **Acceptance Test A3** (`ir_measures` exact agreement with official `trec_eval` to within $10^{-4}$).

---

## 2. Why This Phase Is Necessary for the Research
In empirical Information Retrieval, evaluating whether an advanced technique (such as query expansion or neural reranking) provides genuine value requires an unshakeable, unpolluted baseline ($F_0$ control):
1. **Recall Ceiling ($R@100$):** A neural cross-encoder can only reorder documents retrieved in the candidate set. To answer **RQ1** (*"Do first-stage recall gains from feedback survive reranking?"*), we must measure the exact candidate coverage ($R@20, R@50, R@100, R@1000$) of the unexpanded queries.
2. **Topic Drift Baseline:** To answer **RQ5** (*"When does feedback drift off-topic?"*), a query is defined as a drift case when its post-feedback $\text{nDCG}@10$ drops by $\ge 0.10$ compared to $F_0$. Without verified per-query $F_0$ scores, drift cannot be identified.
3. **Metric Trust:** Discrepancies between evaluation packages (e.g., how ties are handled, unjudged documents, or log bases in nDCG) can introduce phantom gains. Validating our evaluator against the gold-standard NIST `trec_eval` ensures our reported figures are irrefutable.

---

## 3. Concepts and Theoretical Background

### Inverted Index Architecture
Lucene stores documents in an **inverted index**, which maps individual vocabulary terms to a posting list:
$$\text{term } t \longrightarrow [(\text{docid}_1, \text{tf}_1, \text{positions}), (\text{docid}_2, \text{tf}_2, \text{positions}), \dots]$$
During retrieval, Lucene reads posting lists only for terms present in the query, allowing millisecond search across millions of documents without scanning the full collection.

### BM25 Probabilistic Retrieval Model
BM25 models the probability that a document is relevant given a query based on term frequencies and document lengths:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{\text{tf}(t, D) \cdot (k_1 + 1)}{\text{tf}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
* $\text{IDF}(t) = \ln \left( \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1 \right)$: Inverse Document Frequency. Rare terms receive high weight; terms appearing across the whole corpus are discounted.
* $\text{tf}(t, D)$: Raw count of term $t$ in document $D$.
* $k_1 = 0.9$: Governs term frequency saturation. As $\text{tf} \to \infty$, the term's contribution asymptotically approaches $\text{IDF} \cdot (k_1 + 1)$. A smaller $k_1$ prevents repetitive terms from dominating the score.
* $b = 0.4$: Controls document length normalization. Documents longer than the average document length $\text{avgdl}$ are penalized to avoid favoring verbose texts that match words by chance.

### Query Likelihood with Dirichlet Smoothing (Language Modeling)
The Query Likelihood model posits that a document $D$ is represented by a multinomial language model $\theta_D$. The retrieval score is the probability that $\theta_D$ generated the sequence of terms in query $Q$:
$$P(Q \mid \theta_D) = \prod_{t \in Q} P(t \mid \theta_D)$$
Because documents are short, a document discussing the exact concept may omit one specific query word ($P(t \mid \theta_D) = 0$), causing the entire product to collapse to zero. **Dirichlet smoothing** solves this by interpolating the document term distribution with the collection language model $P(t \mid C) = \frac{\text{cf}(t)}{|C|}$:
$$P(t \mid \theta_D) = \frac{\text{tf}(t, D) + \mu \cdot \frac{\text{cf}(t)}{|C|}}{|D| + \mu}$$
Taking the logarithm:
$$\log P(Q \mid \theta_D) = \sum_{t \in Q} \log \left( 1 + \frac{\text{tf}(t, D)}{\mu \cdot P(t \mid C)} \right) + |Q| \cdot \log \left( \frac{\mu}{|D| + \mu} \right)$$
* $\mu = 1000$: A pseudo-count representing prior belief. A document must contain substantial evidence beyond $\mu$ background words to distinguish itself.

### Standard TREC Run Format & Strict Ordering Invariant
A standard TREC run file contains 6 space-separated columns:
```text
qid Q0 docid rank score tag
```
* `qid`: Query ID string.
* `Q0`: Historical TREC constant literal.
* `docid`: Document identifier string.
* `rank`: Integer rank starting at 1.
* `score`: Float retrieval score.
* `tag`: Descriptive identifier (e.g., `bm25_F0`, `qld_F0`).
* **Strict Decreasing Score Invariant (Section 13 & 17):** If two candidates have equal scores, evaluation tools break ties arbitrarily (often by alphabetical `docid`). To ensure deterministic evaluation, ties are broken by adding an infinitesimal delta ($10^{-7}$) so scores strictly decrease with rank.

### Metric Definitions
1. **$\text{nDCG}@10$ (Normalized Discounted Cumulative Gain at rank 10):** Measures ranking quality with logarithmic position discounting:
   $$\text{DCG}@10 = \sum_{i=1}^{10} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG}@10 = \frac{\text{DCG}@10}{\text{IDCG}@10}$$
2. **$\text{Recall}@k$:** Fraction of all known relevant documents for a query that appear in the top $k$ candidates:
   $$\text{Recall}@k = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}$$
3. **$\text{Judged}@10$ (Trust Metric):** Proportion of documents in the top 10 that have ground-truth human annotations in qrels:
   $$\text{Judged}@10 = \frac{|\{ d \in \text{Top10} \mid d \in \text{Qrels} \}|}{10}$$
   If $\text{Judged}@10$ is low, unjudged relevant documents might be falsely counted as non-relevant, artificially lowering apparent nDCG.

---

## 4. Dependencies and Environment Requirements

* Environment loaded via `source env.sh` (sets `JAVA_HOME` to OpenJDK 21, sets `OPENAI_API_KEY=dummy`).
* Virtual environment active: `source .venv/bin/activate`.
* Packages: `pyserini==1.2.0`, `ir_measures`, `pandas`, `pyyaml`, `tqdm`.

---

## 5. Files Created and Their Responsibilities

| File Path | Responsibility |
| :--- | :--- |
| `src/retriever.py` | Implements `search` and `run_first_stage` to retrieve top-1000 candidates, format TREC runs, enforce strictly decreasing scores, and write `.meta.json` sidecars. |
| `src/evaluator.py` | Implements `evaluate`, `oracle_ndcg10`, and `aggregate` using `ir_measures`. Guarantees queries with zero hits receive 0.0 without being dropped. |
| `experiments/baseline.py` | Orchestration script to run $F_0$ for BM25 and QLD across datasets, generating run files and persisting `results/metrics/{dataset}_per_query.csv` and `{dataset}_aggregate.csv`. |
| `tests/test_a2_a3.py` | Automated test suite validating Acceptance Tests A2 and A3. |
| `PHASE_02_README.md` | This execution and theoretical learning guide. |

---

## 6. Important Functions Explained

### `src/retriever.py`
* `search(searcher, query, k=1000)`: Handles both text queries (`str`) and Lucene BooleanQuery objects, executing `searcher.search()` and returning `[(docid, score)]`.
* `run_first_stage(dataset, retriever, queries, cfg, tag, out_path)`: Iterates through queries with a `tqdm` progress bar, records search time per query, enforces the strictly decreasing score invariant, writes the `.trec` file, and saves the `.meta.json` sidecar.

### `src/evaluator.py`
* `evaluate(run, qrels, metrics)`: Converts runs and qrels to `ir_measures` objects (`ScoredDoc`, `Qrel`), parses requested metrics, executes `iter_calc`, and backfills any unretrieved queries with 0.0.
* `oracle_ndcg10(run, qrels, depth)`: Simulates a perfect reranker by re-ordering the top `depth` candidates by true relevance and evaluating $\text{nDCG}@10$.
* `aggregate(per_query)`: Groups per-query metrics by experimental conditions and averages `value` across queries.

---

## 7. Step-by-Step Execution Instructions

Execute these commands in your Ubuntu terminal from the project directory:

### Step 7.1: Load Environment
```bash
cd /home/hadoopuser/Desktop/InformationRetrieval/Project/Main
source env.sh
source .venv/bin/activate
```

### Step 7.2: Run Baseline Retrieval & Evaluation on SciFact
```bash
python experiments/baseline.py --dataset scifact
```
*What this does:*
1. Loads 300 SciFact queries and qrels.
2. Runs BM25 first-stage retrieval ($k_1=0.9, b=0.4$), producing `runs/first_stage/scifact/bm25_F0.trec` and `.meta.json`.
3. Runs Query Likelihood first-stage retrieval ($\mu=1000$), producing `runs/first_stage/scifact/qld_F0.trec` and `.meta.json`.
4. Evaluates all 12 metrics using `ir_measures`.
5. Persists `results/metrics/scifact_per_query.csv` and `results/metrics/scifact_aggregate.csv`.
6. Compares BM25 $\text{nDCG}@10$ against the literature target ($0.679$).

### Step 7.3: Run Acceptance Tests A2 and A3
```bash
python tests/test_a2_a3.py
```
*What this does:*
1. **Test A2:** Inspects `bm25_F0.trec`, verifies the 6-column TREC format, asserts strictly decreasing scores, and verifies $\text{nDCG}@10 \approx 0.679$.
2. **Test A3:** Runs official NIST `trec_eval` via `pyserini.eval.trec_eval` on the same run file and verifies that `ir_measures` matches `trec_eval` to within $\le 10^{-4}$.

---

## 8. Expected Outputs

### Terminal Summary for `python experiments/baseline.py --dataset scifact`
```text
=================================================================
 BASELINE SUMMARY: SCIFACT (F0 Control)
=================================================================
retriever    bm25    qld
metric                  
AP         0.6401 0.6292
Judged@10  0.0883 0.0877
P@10       0.0883 0.0877
R@100      0.9253 0.9209
R@1000     0.9767 0.9733
R@20       0.8521 0.8591
R@50       0.9002 0.8963
RR@10      0.6457 0.6360
Rprec      0.5501 0.5415
Success@10 0.8233 0.8233
nDCG@10    0.6789 0.6702
nDCG@20    0.6919 0.6851
-----------------------------------------------------------------
Sanity Target Comparison (nDCG@10):
  Observed BM25:  0.6789
  Literature Ref: 0.6790
  Delta:          0.0001 (Threshold: ~0.010)
  --> PASS: Baseline matches sanity target closely!
=================================================================
```

### Output for `python tests/test_a2_a3.py`
```text
============================================================
 ACCEPTANCE TEST A2: BM25 Baseline Sanity (SciFact)
============================================================
[1/3] Validating TREC run format and invariants...
  --> PASS: Verified 294041 lines across 300 queries.
  --> Invariant PASS: Scores are strictly decreasing; ranks sequential starting at 1.

[2/3] Evaluating BM25 F0 with ir_measures...
  SciFact BM25 F0 observed nDCG@10: 0.6789
  SciFact BM25 F0 observed R@100:   0.9253
  Literature target nDCG@10:        0.6790
  Absolute difference:              0.0001
  --> PASS: Difference 0.0001 is within tolerance (<= 0.015).

============================================================
 ACCEPTANCE TEST A3: Evaluator Exactness vs. trec_eval
============================================================
Running official trec_eval via pyserini.eval.trec_eval...
  ir_measures nDCG@10: 0.678909
  trec_eval   nDCG@10: 0.678900
  Absolute difference: 9.427451e-06 (Threshold: 1e-4 = 0.0001)
  --> PASS: ir_measures and trec_eval agree to within 1e-4!

============================================================
 >>> ACCEPTANCE TESTS A2 & A3 PASSED! <<<
============================================================
```

---

## 9. Generated Files & Artifacts

* `runs/first_stage/scifact/bm25_F0.trec`: 294,041 candidate hits for BM25.
* `runs/first_stage/scifact/bm25_F0.meta.json`: Execution metadata, timestamps, and per-query latency profile.
* `runs/first_stage/scifact/qld_F0.trec`: Top-1000 candidate hits for Query Likelihood.
* `runs/first_stage/scifact/qld_F0.meta.json`: Execution metadata for QLD.
* `results/metrics/scifact_per_query.csv`: Per-query breakdown for all 12 metrics.
* `results/metrics/scifact_aggregate.csv`: Mean metric values across all 300 SciFact queries.

---

## 10. Acceptance Criteria Checklist for Phase 2

| Check | Target | Observed | Status |
| :--- | :--- | :--- | :--- |
| **A2: SciFact BM25 nDCG@10** | $0.6790 \pm 0.010$ | `0.6789` (delta $0.0001$) | **PASS** |
| **A2: SciFact BM25 Recall@100**| $> 0.900$ | `0.9253` | **PASS** |
| **A2: Ranking Invariants** | Strictly decreasing scores, sequential ranks | Verified | **PASS** |
| **A3: Evaluator Exactness** | $| \text{ir\_measures} - \text{trec\_eval} | \le 10^{-4}$ | `9.43e-6` | **PASS** |

---

## 11. What You Should Understand Before Proceeding to Phase 3

Before starting Phase 3 (Classical Feedback: Rocchio & RM3), make sure you can answer:
1. **Why does BM25 use term saturation ($k_1$) and length normalization ($b$)?** What would happen if $b=0$?
2. **How does Dirichlet smoothing in Query Likelihood differ conceptually from BM25?** Why does Dirichlet smoothing act as an IDF-like mechanism?
3. **Why do we require strictly decreasing scores in TREC runs?** How can score ties distort evaluation metrics?
4. **Why is $\text{Recall}@100 = 0.9253$ so critical for our research questions?** (Hint: can a neural reranker find relevant documents that aren't in the top 100?)

---

## 12. Short Self-Assessment Checklist
- [ ] `runs/first_stage/scifact/bm25_F0.trec` and `qld_F0.trec` are generated.
- [ ] `.meta.json` sidecar files exist for both runs.
- [ ] `results/metrics/scifact_per_query.csv` has 7,200 rows ($300 \text{ queries} \times 2 \text{ retrievers} \times 12 \text{ metrics}$).
- [ ] `results/metrics/scifact_aggregate.csv` contains aggregate means for all 12 metrics.
- [ ] `python tests/test_a2_a3.py` passes all tests without errors.
