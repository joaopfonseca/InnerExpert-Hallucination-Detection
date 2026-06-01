# Baselines

## Selected Baselines

### Must Include

#### 1. Predictive Entropy

The simplest uncertainty baseline: compute the entropy of the model's predicted token probability distribution at each position. High entropy → high uncertainty. If we can't beat this, nothing else matters. Dead simple to implement, sets the floor.

**How it works:** For each token position t, compute H_t = −Σ_v p(v | x_{<t}) log p(v | x_{<t}), then aggregate across the answer (typically mean or max). The aggregated score serves as the uncertainty estimate — higher entropy means the model is less confident about what comes next, suggesting a higher likelihood of hallucination.

**Paper:** Standard entropy-based uncertainty (no single paper; see e.g., Malinin & Gales, 2020)

**Rhetorical purpose:** Floor — can you beat the simplest approach?

#### 2. Semantic Uncertainty (Kuhn et al., 2023 / Nature 2024)

The current gold standard for training-free hallucination detection. Published in *Nature* — every reviewer will look for this. Our most direct competitor: both are uncertainty-based methods, but ours is cheaper (no multiple generations + semantic clustering needed).

**How it works:**

1. **Sample N responses** per question (e.g., 5–10 with temperature > 0).
2. **Cluster by semantic equivalence** — use an NLI model (typically DeBERTa-v3-large fine-tuned on MNLI) to check whether each pair of responses entails each other. Responses that mutually entail are placed in the same cluster. "Paris" and "the French capital" → same cluster; "Paris" and "Lyon" → different clusters.
3. **Compute cluster probabilities** — each cluster's probability is the proportion of sampled responses that fall into it: p(c) = |c| / N.
4. **Compute Semantic Entropy** — entropy over the cluster distribution:

   SE = −Σ_c p(c) log p(c)

**When SE is low:** Most responses cluster together → model is confident (even if worded differently). "Paris" and "France's capital" give SE ≈ 0.

**When SE is high:** Responses split across multiple clusters → model is uncertain → likely hallucinating.

**Key insight:** By clustering at the semantic level rather than the token level, you get a much cleaner uncertainty signal. Regular predictive entropy treats "Paris" and "the French capital" as different answers, inflating uncertainty estimates when the model is actually confident — it just expresses the same answer differently. Semantic clustering fixes this.

**Limitation (addressed by Semantic Energy):** When *all* responses cluster together, SE = 0 regardless of how confident the model actually is in those tokens. The model could be generating with very low logit magnitudes (unsure) or very high logit magnitudes (confident) — SE can't tell the difference. Semantic Energy fixes this by looking at logit magnitudes.

**Paper:** *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation* (ICLR 2023) and *Detecting Hallucinations in Large Language Models Using Semantic Entropy* (Nature, 2024)

**Rhetorical purpose:** Ceiling for training-free — can you match/beat the best without multiple generations?

#### 3. SelfCheckGPT (Manakul et al., EMNLP 2023)

Detects hallucinations by measuring **consistency across multiple sampled generations**. The core idea: if the model gives the same answer across different samples, it "knows" the answer. If answers contradict each other, the model is guessing.

**How it works:**

1. **Sample N responses** per question with temperature > 0
2. **Split responses into sentences**
3. **For each sentence**, check consistency against all other responses:
   - **NLI variant:** Use a natural language inference model (e.g., DeBERTa-v3-large MNLI) to check if each sampled passage supports, contradicts, or is neutral to the target sentence
   - **Prompt variant:** Ask an LLM (GPT-3.5/GPT-4) to judge whether the target sentence is supported by the sampled passages
4. **Aggregate** consistency scores across sentences → answer-level score

The paper proposes five variants (BERTScore, QA, n-gram, NLI, Prompt). **Prompt** achieves the best performance (AUC-PR 93.42) but requires an LLM call. **NLI** offers the best performance-computation tradeoff (AUC-PR 92.50). We implement both as the two most relevant variants.

**Paper:** *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models* (EMNLP 2023)
**Code:** https://github.com/potsawee/selfcheckgpt (pip: `selfcheckgpt`)

**Rhetorical purpose:** Cost argument — same detection quality, fraction of the inference cost.

#### 4. LLM-Check (Mitchell et al., 2023)

Uses hidden state covariance structure (SVD-based scores) and attention patterns to estimate confidence. Training-free, single-pass. We're already implementing their hidden state + attention scores — this is the most natural ablation point to show that adding MoE signals on top strictly improves over standard internal signals alone.

**Important implementation note:** The LLM-Check paper defines the Hidden Score as the mean log-determinant of the uncentered covariance HᵀH. However, the official implementation adds data centering (J = I - (1/m)11ᵀ) and αI regularization. The paper explicitly contrasts this with INSIDE (Chen et al., 2024), which computes a centered covariance across *multiple model responses*. The centering in LLM-Check is applied within a single response's hidden states. We follow the implementation (with centering and regularization) as it produces the reported results and is more numerically stable.

**Paper:** *LLM-Check: Investigating Detection of Hallucinations in Large Language Models* (NeurIPS 2024)
**Code:** https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection

**Rhetorical purpose:** Ablation — do MoE signals add value on top of standard internal signals?

### Should Include

#### 5. Semantic Energy (Ma et al., 2025)

Addresses semantic entropy's weakness by operating on penultimate-layer logits with a Boltzmann-inspired energy function instead of post-softmax probabilities. Recent, captures model uncertainty in cases where semantic entropy fails. Related to our internal signal approach — good to show that routing signals capture something logit-space energy doesn't.

**Paper:** *Semantic Energy: Detecting LLM Hallucination Beyond Entropy* (arXiv 2508.14496)

**Rhetorical purpose:** Internal signal comparison — do routing signals beat logit-space energy?

#### 6. HaluNet (Tong et al., 2025)

A lightweight trainable multi-branch neural framework that fuses three token-level uncertainty signals:

1. **Log-likelihood branch** — per-token log probabilities (probabilistic confidence)
2. **Entropy branch** — per-token predictive entropy (distributional uncertainty)
3. **Embedding branch** — hidden-state embeddings via 2-layer 1D Conv (semantic trajectory)

Each branch produces a latent vector; branch outputs are fused via attention or MLP, then projected to a single hallucination probability. Trained with binary cross-entropy on LLM-as-a-Judge labels.

**Architecture:** Scalar features (log-likelihoods, entropies) → mean pooling + 2-layer MLP; embedding features → 2-layer Conv1D + ReLU + adaptive avg pooling; attention-based or MLP fusion; output sigmoid.

**Key distinction from our method:** HaluNet uses standard transformer signals (logits, hidden states) across all layers/tokens. Our method adds MoE-specific routing signals (router entropy, expert disagreement, Gini, Herfindahl) on top of these, providing complementary uncertainty information from the routing behavior itself. **Both methods are trained on the same train/val split** (train for fitting, val for early stopping / threshold tuning), making this a fair comparison of feature sets within the same training paradigm.

**Paper:** *HaluNet: Multi-Granular Uncertainty Modeling for Efficient Hallucination Detection in LLM Question Answering* (arXiv 2512.24562)

**Rhetorical purpose:** Trainable comparison — both our method and HaluNet are lightweight trainable classifiers. Can MoE signals improve over standard signals within the same training paradigm?

### 7. Token-Level Mahalanobis Distance (Vazhentsev et al., 2025)

A density-based uncertainty quantification method adapted from classification OOD detection to text generation. Instead of computing uncertainty from output probabilities or generation consistency, it looks at the geometry of hidden state embeddings across decoder layers.

**How it works:**

1. **Extract token embeddings** from multiple decoder layers (not just the last layer).
2. **Estimate density** of the "factual" embedding distribution: for each layer, compute the class-conditional mean and shared covariance of tokens labeled as factual (ground truth).
3. **Compute Mahalanobis distance** per token per layer: MD(x) = √((x − μ)ᵀ Σ⁻¹ (x − μ)). This measures how far a token's embedding is from the factual distribution — larger MD → more atypical → more likely hallucinated.
4. **Dimensionality reduction**: Apply PCA across the layer-wise MD features.
5. **Train linear regression** (Ridge) on the PCA-reduced features, optionally augmented with the sequence's log-probability, to produce a continuous uncertainty score.

**Key properties:**
- **Per-token:** YES — computes an uncertainty score for each generated token.
- **Generations needed:** 1 (single-pass, no sampling).
- **Training required:** YES — requires labeled "factual" tokens to estimate class-conditional means and covariance, plus regression training.
- **Signals used:** Hidden state embeddings from multiple decoder layers (+ optional log-probabilities).
- **OOD generalization:** Strong, since Mahalanobis distance measures distributional atypicality regardless of the specific task.
- **Computational efficiency:** Moderate — requires per-layer covariance estimation (O(d³) per layer where d = hidden size) and PCA. For typical setups (d=768, ~20-32 layers), this is tractable.

**Paper:** Vazhentsev et al., "Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models" (arXiv 2502.14427, 2025)
**Code:** https://github.com/ArtemVazh/token_mahalanobis_distance

**Rhetorical purpose:** Internal signal comparison — does MoE routing beat density-based uncertainty from hidden states?

### 8. TOHA — TOpology-based HAllucination detector (Bazarova et al., 2025)

A fundamentally different approach that treats attention maps as weighted graphs and uses topological data analysis to detect hallucination. Completely orthogonal to both routing-based and probability-based methods.

**How it works (original):**

1. **Build attention graphs**: For each attention head, treat the attention matrix A ∈ ℝ^(seq_len × seq_len) as a weighted directed graph. The prompt tokens form one subgraph; the generated tokens form another.
2. **Compute topological divergence**: Use persistent homology to compare the topological structure of prompt and response subgraphs. Persistent homology tracks how connected components, loops, and voids in the graph appear and disappear as the filtration threshold varies.
3. **Identify hallucination-aware heads**: Some attention heads exhibit systematic topological differences between factual and hallucinated responses. These heads can be selected with minimal annotated data.
4. **Aggregate divergence**: The final score is the aggregate topological divergence across selected heads.

**Practical approximation (this implementation):**

Since persistent homology requires specialized TDA libraries (gudhi, ripser) and is computationally expensive, we implement a practical approximation based on attention entropy:

1. **Attention entropy per token**: For each head, compute the Shannon entropy of each token's attention distribution. High entropy = attention is spread diffusely across many source tokens; low entropy = attention is sharply focused.
2. **Prompt-to-response entropy shift**: Measure the change in attention entropy from prompt tokens to generated tokens. Hallucinated responses tend to have more diffuse attention (higher entropy) on response tokens.
3. **Concentration ratio**: Fraction of attention mass on the top-K source tokens. Hallucinated responses have lower concentration (more distributed attention).
4. **KL divergence**: Divergence between prompt attention distribution and response attention distribution. High divergence indicates the response is attending to different patterns.
5. **Frobenius divergence**: Norm of the difference between the prompt and response attention subgraphs.
6. **Spectral entropy**: Entropy of the graph Laplacian's eigenvalue distribution, capturing overall topological complexity.

These features are combined per head; the fit() method selects heads that best discriminate hallucinated vs factual responses via AUROC-based weighting.

**Key properties:**
- **Per-token:** NO (head-level, can be mapped to tokens).
- **Generations needed:** 1 (single-pass).
- **Training required:** NO (training-free head selection, though few annotated examples improve selection).
- **Signals used:** Attention matrices (all layers, all heads).
- **Computational efficiency:** Lightweight — feature extraction is O(seq_len² × n_layers × n_heads), comparable to computing attention metrics.
- **Orthogonal signal:** Attention topology captures completely different structure from routing entropy or output probability.

**Paper:** Bazarova et al., "Hallucination Detection in LLMs with Topological Divergence on Attention Graphs" (arXiv 2504.10063, 2025)
**Code:** https://github.com/sb-ai-lab/TOHA

**Rhetorical purpose:** Orthogonal signal — does topology of attention capture uncertainty that routing and density miss?

## Excluded Baselines

| Baseline | Reason for Exclusion |
|---|---|
| FacLens | Not well-cited enough; no clean implementation; poor effort-to-value ratio |
| In-Context Confidence Prompting | Too weak and unreliable; reviewers won't care if we beat it |
| DegLM | Originally for density estimation in smaller models; not established for LLM hallucination detection; invites "why this baseline?" questions |

## Baseline Matrix

| Baseline | Type | Generations Needed | Training Required | Signals Used |
|---|---|---|---|---|
| Predictive Entropy | Entropy-based | 1 | No | Output probabilities |
| Semantic Uncertainty | Generation-based | Multiple | No | Semantic clusters of generations |
| SelfCheckGPT | Generation-based | Multiple | No | Cross-generation consistency |
| LLM-Check | Internal signal | 1 | No | Hidden states, attention |
| Semantic Energy | Internal signal | 1 | No | Penultimate logits + semantic clustering |
| HaluNet | Trainable | 1 | Yes | Token probs, semantic embeddings, distributional uncertainty |
| Token Mahalanobis | Density-based | 1 | Yes | Hidden states (multi-layer) |
| TOHA | Topology-based | 1 | No | Attention matrices |
| **Ours** | Internal signal (MoE) | 1 | No (optionally) | All of LLM-Check + routing entropy, expert similarity, expert usage, Gini, Herfindahl |