# Proposed Method

## Core Idea

Mixture-of-Experts (MoE) language models expose internal routing signals that dense models do not provide. When an MoE model processes a token, a gating network selects a sparse subset of experts — and the pattern of that selection (which experts, how confidently, how consistently) carries information about the model's uncertainty. Our method leverages these MoE-specific signals, alongside standard internal signals (hidden states, attention, output entropy), to detect hallucinations at both the answer level and the token level.

The key hypothesis: **MoE routing signals provide complementary uncertainty information that improves hallucination detection beyond what standard signals alone can achieve.**

## Method Overview

### 1. Signal Extraction

Using the `MoEMonitor` wrapper, we intercept the model's forward pass to collect a comprehensive set of signals at each generation step:

**Standard signals (available in any transformer):**
- **Hidden state scores** — SVD-based confidence from hidden state covariance matrices (from LLM-Check)
- **Attention scores** — diagonal log-cumulative sum of attention matrices (from LLM-Check)
- **Top-k output entropy** — entropy over the model's predicted token distribution

**MoE-specific signals:**
- **Router entropy** — entropy of the gating distribution across experts; high entropy indicates the model is uncertain about which expert should handle the token
- **Expert hidden scores** — weighted hidden state score across selected experts, capturing whether individual experts themselves are confident
- **Expert similarity scores** — routing-weighted cosine similarity between expert hidden states; low similarity suggests experts disagree, indicating uncertainty. This is our primary proxy for epistemic uncertainty: expert disagreement in representation space is analogous to ensemble disagreement, which is a well-established epistemic uncertainty measure (cf. Pavlitska et al., 2025, who use expert variance — inversely related to cosine similarity — for the same purpose)
- **Expert usage frequency** — how often each expert is selected across the sequence, normalized cumulatively
- **Gini impurity of expert usage** — concentration of expert selection; low impurity means few experts dominate, high impurity means routing is scattered
- **Inverse Herfindahl index (effective number of experts)** — captures how many experts are effectively contributing

### 2. Signal Aggregation

The extracted signals are collected per token position and per layer. The goal is to produce **per-token epistemic uncertainty estimates** — a confidence score for each generated token reflecting how likely it is to be hallucinated.

- **Token-level uncertainty** — the primary output of our method; per-position epistemic uncertainty proxy derived from routing and standard signals
- **Answer-level scores** — used as an intermediate step to generate training and test labels (e.g., by aggregating token-level predictions or using external metrics like ROUGE/BLEU against references), not as the final goal

### 3. Hallucination Detection

Signals are combined to predict per-token hallucination labels:

- **Training-free approach** — use individual signals or simple combinations (e.g., logistic regression) as uncertainty scores, with thresholds optimized on validation data
- **Trainable approach** — train a lightweight classifier (logistic regression, small MLP) on labeled data (from weak labels + LLM labeling) using the per-token signal vector as features; answer-level labels serve as supervision signal during training

## What Makes This Different

1. **Post-hoc** — requires no modification to the model architecture or training; extracts signals from the existing routing behavior of any MoE model
2. **Complementary** — MoE signals capture a different dimension of uncertainty than output probability or hidden state analysis alone; they reflect *which knowledge domains the model is activating*, not just *how confident the output distribution is*
3. **Fine-grained** — routing signals are available per-token and per-layer, enabling token-level hallucination detection rather than just answer-level
4. **Architecture-aware** — exploits the specific structure of MoE models rather than treating them as black boxes; this is information that is simply unavailable in dense models

## Expected Advantages over Baselines

- vs. **SelfCheckGPT / Semantic Uncertainty**: our method is single-pass (no need for multiple generations), significantly cheaper at inference time
- vs. **LLM-Check**: we add MoE-specific signals on top of their hidden state + attention approach, providing strictly more information
- vs. **Semantic Energy**: we incorporate routing-level uncertainty rather than relying solely on logit-space energy, capturing structural uncertainty the penultimate layer may not reflect
- vs. **HaluNet**: our method is training-free (in its simplest form), whereas HaluNet requires training a multi-branch neural network
- vs. **Predictive Entropy**: router entropy captures *routing* uncertainty, which is orthogonal to output entropy and can detect cases where the model produces a confident output despite uncertain routing

## Connection to Epistemic Uncertainty

Our MoE routing signals can be interpreted as **proxies for epistemic uncertainty** — uncertainty arising from the model's lack of knowledge about the ground truth — rather than formal measurements of it. This distinction matters, but the connection is well-supported:

**Theoretical grounding:**
- **Expert disagreement as epistemic uncertainty.** In the ensemble uncertainty literature, disagreement among ensemble members is a standard proxy for epistemic uncertainty (Depeweg et al., 2018). An MoE model's experts can be viewed as an implicit ensemble, and their disagreement (low similarity, high mutual information) serves the same role. Pavlitska et al. (2025) explicitly demonstrate this connection for MoE models in semantic segmentation, showing that mutual information between experts captures epistemic uncertainty and outperforms traditional ensembles for OoD detection.
- **Epistemic uncertainty → hallucination.** Yadkori et al. (2024) formalize the relationship between epistemic uncertainty and hallucinations in LLMs, showing that when epistemic uncertainty is high, the model's output is unreliable — i.e., likely hallucinated. This gives a theoretical basis for using epistemic uncertainty proxies as hallucination detectors.
- **Router entropy as model uncertainty.** High routing entropy means the gating network cannot confidently assign a token to a specific expert, suggesting the input falls in a region where the model's knowledge is insufficient — structurally analogous to epistemic uncertainty.

**Important caveats:**
- Our signals come from a **single deterministic forward pass**, not from a Bayesian posterior or an explicit ensemble. Traditional epistemic uncertainty quantification requires some notion of "what the model would do under different parameterizations." Router entropy from one pass is a proxy, not a formal measure.
- Some signals (output entropy, expert hidden scores) **mix epistemic and aleatoric** uncertainty and cannot cleanly separate the two from a single pass.
- Taparia et al. (2026) argue that the classical epistemic/aleatoric dichotomy is insufficient for LLMs and propose a three-way decomposition (input ambiguity, knowledge gaps, decoding randomness). Our MoE signals most closely align with their "knowledge gaps" component — when the model lacks parametric evidence for a domain, routing becomes uncertain.

**Practical implication:** We frame our method as providing *proxies for epistemic uncertainty* rather than formally measuring it. This is consistent with how most practical hallucination detection methods operate — even Semantic Uncertainty (Kuhn et al., 2023) uses semantic entropy as a proxy, not a direct epistemic measure. The key empirical question is whether MoE routing proxies are *better* proxies than existing ones, which our experiments will address.

## Open Questions

- Which MoE signals are most discriminative for hallucination detection? (Ablation needed)
- Do MoE signals generalize across different MoE architectures (OLMoE, Mixtral, DeepSeek, Granite)?
- Can token-level routing signals reliably identify hallucinated spans, or is answer-level detection the practical limit?
- How do MoE signals interact with standard signals — are they truly complementary or largely redundant?