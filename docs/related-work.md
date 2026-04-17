# Related Work

Methods for hallucination detection and uncertainty estimation in LLMs, organized by category.

---

## Generation-Based Methods

These methods require sampling multiple generations from the model and analyzing their consistency.

### SelfCheckGPT (Manakul et al., 2023)

Detects hallucinations by measuring consistency across multiple sampled generations. If the model produces contradictory answers across samples, it likely doesn't "know" the answer. Uses BERTScore or n-gram overlap between generations as a consistency metric. Training-free and model-agnostic, but expensive (requires multiple generations per query).

**Paper:** *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models* (EMNLP 2023)

### Semantic Uncertainty (Kuhn et al., 2023)

Extends predictive entropy by clustering generations by semantic equivalence rather than exact string match. Computes entropy over semantic clusters of multiple sampled responses, which better captures the model's uncertainty when it expresses the same idea in different ways. State-of-the-art on several benchmarks at time of publication.

**Paper:** *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation* (ICLR 2023)

---

## Internal Signal Methods

These methods leverage the model's internal activations (hidden states, attention, logits) rather than or in addition to output text.

### LLM-Check (Mitchell et al., 2023)

Uses hidden state covariance structure and attention patterns to estimate confidence. Computes singular value decomposition of hidden state covariance matrices across layers — the cumulative log of singular values serves as a confidence score. Also uses attention diagonal log-cumulative sums. Training-free, single-pass. **Note:** Our `hidden_score` and `attention_score` metrics are directly based on this method.

**Paper:** *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature* and related work on internal state analysis for hallucination detection.

### FacLens (Chen et al., 2024)

Focuses on factuality detection by analyzing the model's internal layer-wise representations. Identifies "factual" vs. "non-factual" generations by examining how information flows through transformer layers. Operates on hidden states without requiring multiple generations.

### Semantic Energy (Ma et al., 2025)

Addresses a key weakness of semantic entropy: it operates on post-softmax probabilities, which lose uncertainty signal through the softmax normalization. Instead, Semantic Energy uses a Boltzmann-inspired energy function on the penultimate layer logits combined with semantic clustering. Better captures model uncertainty in cases where semantic entropy fails. Evaluated across multiple benchmarks with significant improvements.

**Paper:** *Semantic Energy: Detecting LLM Hallucination Beyond Entropy* (arXiv 2508.14496)

### HaluNet (Tong et al., 2025)

A lightweight, trainable neural framework that fuses multi-granular uncertainty signals: token-level probability uncertainty, semantic embeddings, and distributional uncertainty. Its multi-branch architecture adaptively combines what the model "knows" (semantic representations) with how uncertain its outputs are. Enables efficient one-pass hallucination detection. Evaluated on SQuAD, TriviaQA, and Natural Questions with and without context access.

**Paper:** *HaluNet: Multi-Granular Uncertainty Modeling for Efficient Hallucination Detection in LLM Question Answering* (arXiv 2512.24562)

### Unconditional Truthfulness (Vazhentsev et al., EMNLP 2025)

Learns an uncertainty signal that is unconditional — it doesn't depend on specific input features. Rather than estimating per-input uncertainty, it models the model's inherent tendency to produce truthful outputs. Must be combined with human evaluation to quantify alignment with human judgment.

**Paper:** *Unconditional Truthfulness: Learning Unconditional Uncertainty of Large Language Models* (EMNLP 2025)

---




---

## Entropy & Probability Methods

### Predictive Entropy

The simplest uncertainty baseline: compute the entropy of the model's predicted token probability distribution at each position. High entropy → high uncertainty. While straightforward, it often underperforms compared to methods that consider semantic equivalence or internal representations.

### DegLM — Density-based Confidence (Dettmers et al., 2022)

Estimates confidence using density estimation in the model's output probability space. Leverages the idea that low-density regions of the output distribution correspond to uncertain predictions. Provides a non-MoE probability-based baseline.

---

## Prompting-Based Methods

### In-Context Confidence Prompting

Ask the model to self-assess its confidence via prompt engineering (e.g., "How confident are you in this answer? Rate 1-5"). Surprisingly effective in some settings but unreliable in others — models can be miscalibrated about their own confidence. Serves as a lightweight baseline.

---

## Safety Evaluation

### SafeNudge

Evaluates model safety rather than factuality specifically. May be relevant as a complementary evaluation dimension but targets a different problem (harmful outputs vs. factually incorrect outputs).

---

## Uncertainty Decomposition & Epistemic Uncertainty

Methods that decompose LLM uncertainty into epistemic and aleatoric components, or propose alternative decompositions. Relevant for understanding whether MoE routing signals can be interpreted as epistemic uncertainty proxies.

### To Believe or Not to Believe Your LLM (Yadkori et al., 2024)

Formalizes the epistemic/aleatoric split for LLMs and shows that high epistemic uncertainty → hallucination. Derives an information-theoretic metric to detect when only epistemic uncertainty is large (i.e., the output is unreliable) using iterative prompting. Provides theoretical grounding for using epistemic uncertainty proxies as hallucination detectors.

**Paper:** *To Believe or Not to Believe Your LLM* (arXiv 2406.02543)

### Extracting Uncertainty Estimates from MoEs (Pavlitska et al., 2025)

The most directly relevant work for our epistemic uncertainty claim. Shows that uncertainty estimates can be extracted from MoE models without architectural modifications, using predictive entropy, mutual information between experts, and expert variance. Demonstrates that MoE mutual information captures epistemic uncertainty and outperforms ensembles for OoD detection. Conducted in semantic segmentation (not NLP), but the theoretical framework transfers directly.

**Paper:** *Extracting Uncertainty Estimates from Mixtures of Experts for Semantic Segmentation* (arXiv 2509.04816)

### The Anatomy of Uncertainty in LLMs (Taparia et al., 2026)

Argues the classical epistemic/aleatoric dichotomy is insufficient for LLMs. Proposes a three-way decomposition: input ambiguity, knowledge gaps, and decoding randomness. Their "knowledge gaps" component maps closely to what MoE routing signals would capture. Shows that the dominance of these components shifts across model size and task.

**Paper:** *The Anatomy of Uncertainty in LLMs* (arXiv 2603.24967)

---

## Surveys

### UQ for Hallucination Detection (Kang et al., 2025)

Comprehensive survey covering UQ foundations (epistemic vs. aleatoric uncertainty), systematic categorization of existing methods, and empirical results for representative approaches. Good reference for positioning our work.

**Paper:** *Uncertainty Quantification for Hallucination Detection in Large Language Models: Foundations, Methodology, and Future Directions* (arXiv 2510.12040)