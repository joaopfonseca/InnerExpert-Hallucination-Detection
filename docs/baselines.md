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

### 7. Token-Level Mahalanobis Distance (Vazhentsev et al., 2025) — FAITHFUL IMPLEMENTATION

A supervised density-based uncertainty quantification method adapted from classification OOD detection to text generation.

**How it works (exact algorithm from the paper):**

1. **Extract token embeddings** from specified decoder layer(s). The paper supports both single-layer and multi-layer variants.

2. **Estimate density** on "correct/factual" training tokens only. For each layer, compute:
   - **Class-conditional centroid**: mean embedding of all tokens labeled as factual (label=0).
   - **Covariance matrix**: compute Σ = (X − μ)ᵀ (X − μ) / (n−1) where μ is the *known* centroid (not empirical mean — matching the official lm_polygraph implementation).
   - **Regularized inverse**: invert with progressive jitter (1e-6 through 1.0) if singular, falling back to pseudoinverse.

3. **Compute Mahalanobis distance** per token per layer for *all* tokens:
   MD(x) = √( (x − μ)ᵀ Σ⁻¹ (x − μ) )

4. **Sequence-level aggregation**: average MD scores across tokens within each answer to get a per-sequence MD feature.

5. **Ridge regression** (with positive coefficient constraint) on sequence-level MD features against a quality score. The paper uses continuous metrics (F1, correctness) as targets, not binary labels.

6. **HUQ two-stage combination** (Hybrid Uncertainty Quantization): When enabled, MD serves as the *epistemic* uncertainty signal, while Maximum Sequence Probability (MSP) serves as the *aleatoric* signal. These are combined via a ranking-based two-stage formula with parameters (t_min, t_max, α) learned via grid search on a held-out validation split.

**Key details matching the paper's official implementation:**
- Covariance centering uses the fixed centroid μ, not the empirical batch mean.
- Progressive jitter sequence matches lm_polygraph: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0].
- Ridge with `positive=True` (coefficients constrained to be positive, as the paper found best).
- HUQ grid search ranges: t_min ∈ [0.0, 0.3], t_max ∈ [0.7, 1.0], α ∈ [0.0, 1.0].

**Paper:** Vazhentsev et al., "Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models" (NAACL 2025, arXiv 2502.14427)
**Code:** https://github.com/ArtemVazh/token_mahalanobis_distance

**Rhetorical purpose:** Internal signal comparison — does MoE routing beat density-based uncertainty from hidden states?

### 8. TOHA — TOpology-based HAllucination detector (Bazarova et al., 2025) — FAITHFUL IMPLEMENTATION

A fundamentally different approach that treats attention maps as weighted graphs and uses topological data analysis to detect hallucination. Completely orthogonal to both routing-based and probability-based methods.

**How it works (exact algorithm from the paper):**

1. **Build distance matrices**: For each attention head, transform the attention matrix A ∈ ℝ^(seq_len × seq_len) into a distance matrix: `d_ij = 1 − a_ij` (clipped to [0, 1]), zero the diagonal, and symmetrize via `min(d_ij, d_ji)`.

2. **Zero out prompt subgraph**: Set all prompt-to-prompt distances to 0. This isolates the topological structure of the response tokens relative to the prompt. (The paper uses `zero_out="prompt"`; `zero_out="response"` is also supported.)

3. **Compute MTopDiv via persistent homology**: Run Vietoris-Rips filtration using the `ripser` library on the distance matrix (max dimension 0, for connected components). Sum the finite H₀ barcode lengths (birth − death), excluding the infinite component `[0, ∞)`. This sum is the **MTopDiv** (Manifold Topology Divergence) score.

4. **Normalize by response length**: Divide MTopDiv by the number of response tokens, matching the paper's default.

5. **Supervised head selection**: Use `SelectKBest` with ANOVA F-value (`f_classif`) to select the top-n attention heads that best discriminate between hallucinated and factual samples. The paper searches n from 1 to n_max (default 6) and picks the one with highest validation AUROC.

6. **Classification**: Train a `LogisticRegression` on the selected head MTopDiv features. Prediction = `predict_proba[:, 1]`.

7. **Unsupervised mode**: Select heads by difference-of-means between hallucinated and factual MTopDiv scores. Score = mean MTopDiv across selected heads (no classifier).

**Key properties:**
- **Per-token:** NO (head-level, but maps to per-sample scores).
- **Generations needed:** 1 (single-pass).
- **Training required:** Minimal — supervised mode needs labeled examples for head selection; unsupervised mode needs only a few to estimate head ranking.
- **Signals used:** Attention matrices (all layers, all heads).
- **Computational efficiency:** O(seq_len² × n_heads × n_layers) for distance matrix + ripser O(n³) per head in worst case.
- **Orthogonal signal:** Attention topology captures completely different structure from routing entropy or output probability.

**Paper:** Bazarova et al., "Hallucination Detection in LLMs with Topological Divergence on Attention Graphs" (arXiv 2504.10063, 2025) — **ACL 2026**
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