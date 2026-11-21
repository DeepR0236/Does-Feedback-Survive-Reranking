# Project Implementation Notes & Methodological Decisions

This document records methodological decisions, implementation details, and verification checks following the specifications of `Main_Plan_Revised.pdf`.

---

## 1. Term Analysis and Vocabulary Filtering (Section 15.1)
* **Lucene Analyzer:** Porter stemmer with default English stopwords removed (`Analyzer(get_lucene_analyzer(stemmer='porter', stopwords=True))`).
* **Reference Behavior on `get_term_counts`:** As required by Section 15.1, `index_reader.get_term_counts(term)` is called directly on already-analyzed/stemmed terms matching the reference implementations of Jedidi & Lin (2025, 2026). Terms not present in the index or with document frequency ratio $\text{df} / N_{\text{docs}} > 0.1$ are dropped.
* **Alphanumeric Filtering for RM3:** For RM3 on real documents ($F_2$), non-alphanumeric terms are dropped (`term.isalnum()`) to remove artifactual indexing tokens.

---

## 2. RM3 Interpolation Direction Verification (Section 15.4)
* **Relevance Model Formulation:** Follows Abdul-Jaleel et al. (2004) and Lavrenko & Croft (2001).
* **Interpolation Equation:**
  $$w(t) = (1 - \lambda) \cdot q(t) + \lambda \cdot \text{rel}(t)$$
  With `original_query_weight: 0.5` ($\lambda = 0.5$), the query vector $q$ and relevance model $\text{rel}$ receive equal weight ($0.5 \cdot q(t) + 0.5 \cdot \text{rel}(t)$).
* **Validation:** Verified that $\sum_t q(t) = 1.0$ and $\sum_t \text{rel}(t) = 1.0$ before blending, guaranteeing that the combined feedback weights sum to 1.0 and all term weights remain positive.

---

## 3. Query Likelihood Document Score Conversion for Feedback (Page 20 & 21, Acceptance Test A5)
* **Problem:** Raw Query Likelihood scores under Dirichlet smoothing are negative log-likelihoods. Clamping to $10^{-9}$ distorts relative relevance and breaks ranking.
* **Solution (Softmax with $T=1.0$):**
  $$\text{weight}_i = \frac{\exp((s_i - s_{\max}) / 1.0)}{\sum_j \exp((s_j - s_{\max}) / 1.0)}$$
  Subtracting $s_{\max} = \max_i s_i$ ensures numerical stability against underflow/overflow.
* **Verification (A5 Check):**
  - All feedback weights are strictly finite and non-negative.
  - Weights sum to 1.0.
  - Strict rank ordering is preserved ($\text{weight}_i > \text{weight}_{i+1}$ whenever $s_i > s_{i+1}$).
  - Never collapses to accidental uniformity when raw scores differ.
