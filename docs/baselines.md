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

### 7. Token-Level Mahalanobis Distance (Vazhentsev et al., 2025) — **NOT EVALUATED, NOT IN CODEBASE**

A supervised density-based uncertainty quantification method adapted from classification OOD detection to text generation. The most directly comparable density-based internal-signal method to ours, but **we do not include it in our headline evaluation and we do not ship an implementation** — see [§Cited but Not Evaluated Baselines](#cited-but-not-evaluated-baselines) below for the rationale (storage cost is the binding constraint).

**Paper:** Vazhentsev et al., "Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models" (NAACL 2025, arXiv 2502.14427)
**Code:** https://github.com/ArtemVazh/token_mahalanobis_distance

**Rhetorical purpose:** Internal signal comparison — does MoE routing beat density-based uncertainty from hidden states?

### 8. TOHA — TOpology-based HAllucination detector (Bazarova et al., 2025) — **NOT EVALUATED, NOT IN CODEBASE**

A fundamentally different approach that treats attention maps as weighted graphs and uses topological data analysis to detect hallucination. Fully orthogonal to both routing-based and probability-based methods, but **we do not include it in our headline evaluation and we do not ship an implementation** — see [§Cited but Not Evaluated Baselines](#cited-but-not-evaluated-baselines) below for the rationale (storage cost is the binding constraint).

**Paper:** Bazarova et al., "Hallucination Detection in LLMs with Topological Divergence on Attention Graphs" (arXiv 2504.10063, 2025) — **ACL 2026**
**Code:** https://github.com/sb-ai-lab/TOHA

**Rhetorical purpose:** Orthogonal signal — does topology of attention capture uncertainty that routing and density miss?

## Cited but Not Evaluated Baselines

These are recent, well-cited internal-signal detectors that we **cite in the related work** but **do not include in our headline evaluation**, and for which we **do not ship an implementation**. Both are sequence-level detectors that would require us to materialize very large volumes of internal state at inference time just to produce a single per-response score — a storage footprint that is not feasible for our test-time evaluation pipeline. We discuss them in the related work as the closest *sequence-level* cousins of our per-token MoE routing method.

| Baseline | Why we don't evaluate it |
|---|---|
| **Token-Level Mahalanobis Distance** (Vazhentsev et al., NAACL 2025) | **Storage cost is the binding constraint.** The method requires materializing the per-layer hidden-state embeddings of every generated token, plus the per-layer inverse covariance matrices (one `(hidden_dim × hidden_dim)` matrix per layer), at inference time. For a long-context response on a multi-layer MoE, that is `(seq_len × n_layers × hidden_dim)` floats per response *plus* the per-layer inverse covariances — on the order of gigabytes per response. Materializing that much state for every generated response in our test-time pipeline is not feasible. (The method is also inherently sequence-level: the paper's final score is the *mean* Mahalanobis distance across tokens in the response, with a Ridge meta-model trained on top, so it cannot localize hallucinations to specific tokens.) |
| **TOHA** (Bazarova et al., ACL 2026) | **Storage cost + per-response score only.** TOHA requires the **full** attention matrices (all layers, all heads, all token pairs) to build the response-attention subgraph, then runs Vietoris–Rips persistent homology on each head's distance matrix to compute a single MTopDiv scalar per head. For a model with `n_layers × n_heads` heads and a sequence of length `L`, that is `(n_layers × n_heads × L × L)` floats of attention state per response — again on the order of gigabytes for a typical MoE, even after discarding the prompt–prompt subgraph. The resulting score is also a property of the full response attention *graph*; there is no principled way to assign a per-token score from persistent homology applied to the full graph, so the method cannot be evaluated at the token level. |

**Why we cite them anyway.** Both are recent, high-quality, *internal-signal* detectors (the same family our method belongs to) and both use signals that are orthogonal to ours — Mahalanobis uses embedding density, TOHA uses attention topology. We discuss them in the related work as the closest *sequence-level* cousins of our per-token MoE routing method, because they are the natural points of comparison for any future work that *does* have the storage budget to evaluate sequence-level internal-signal detectors. Our headline comparison is restricted to baselines that can be evaluated in the same per-token, low-storage regime as our method.

---

## Excluded Baselines

| Baseline | Reason for Exclusion |
|---|---|
| FacLens | Not well-cited enough; no clean implementation; poor effort-to-value ratio |
| In-Context Confidence Prompting | Too weak and unreliable; reviewers won't care if we beat it |
| DegLM | Originally for density estimation in smaller models; not established for LLM hallucination detection; invites "why this baseline?" questions |

## Baseline Matrix

| Baseline | Type | Generations Needed | Training Required | Per-Token? | Signals Used |
|---|---|---|---|---|---|
| Predictive Entropy | Entropy-based | 1 | No | ✅ Yes | Output probabilities |
| Semantic Uncertainty | Generation-based | Multiple | No | ❌ No (answer-level) | Semantic clusters of generations |
| SelfCheckGPT | Generation-based | Multiple | No | ❌ No (answer-level) | Cross-generation consistency |
| LLM-Check | Internal signal | 1 | No | ✅ Yes | Hidden states, attention |
| Semantic Energy | Internal signal | 1 | No | ❌ No (answer-level) | Penultimate logits + semantic clustering |
| HaluNet | Trainable | 1 | Yes | ⚠️ Trained on token-level, evaluated at answer-level | Token probs, semantic embeddings, distributional uncertainty |
| Token Mahalanobis | Density-based | 1 | Yes | ❌ No — see [Cited but Not Evaluated](#cited-but-not-evaluated-baselines) | Hidden states (multi-layer) |
| TOHA | Topology-based | 1 | No | ❌ No — see [Cited but Not Evaluated](#cited-but-not-evaluated-baselines) | Attention matrices |
| **Ours** | Internal signal (MoE) | 1 | No (optionally) | ✅ Yes | All of LLM-Check + routing entropy, expert similarity, expert usage, Gini, Herfindahl |