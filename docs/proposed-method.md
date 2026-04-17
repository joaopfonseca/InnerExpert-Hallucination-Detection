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
- **Expert similarity scores** — routing-weighted cosine similarity between expert hidden states; low similarity suggests experts disagree, indicating uncertainty
- **Expert usage frequency** — how often each expert is selected across the sequence, normalized cumulatively
- **Gini impurity of expert usage** — concentration of expert selection; low impurity means few experts dominate, high impurity means routing is scattered
- **Inverse Herfindahl index (effective number of experts)** — captures how many experts are effectively contributing

### 2. Signal Aggregation

The extracted signals are collected per token position and per layer. We aggregate them into feature vectors that can be used at two granularities:

- **Answer-level detection** — aggregate signals across all generated tokens (mean, max, variance) to produce a single hallucination confidence score for the entire answer
- **Token-level detection** — use per-position signals to identify which specific tokens are likely hallucinated

### 3. Hallucination Detection

Signals are combined to predict hallucination labels:

- **Training-free approach** — use individual signals or simple combinations (e.g., logistic regression) as uncertainty scores, with thresholds optimized on validation data
- **Trainable approach** — train a lightweight classifier (logistic regression, small MLP) on labeled data (from weak labels + LLM labeling) using the signal vector as features

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

## Open Questions

- Which MoE signals are most discriminative for hallucination detection? (Ablation needed)
- Do MoE signals generalize across different MoE architectures (OLMoE, Mixtral, DeepSeek, Granite)?
- Can token-level routing signals reliably identify hallucinated spans, or is answer-level detection the practical limit?
- How do MoE signals interact with standard signals — are they truly complementary or largely redundant?