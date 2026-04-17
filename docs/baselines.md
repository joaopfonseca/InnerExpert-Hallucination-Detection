# Baselines

## Selected Baselines

### Must Include

#### 1. Predictive Entropy

The simplest uncertainty baseline: compute the entropy of the model's predicted token probability distribution at each position. High entropy → high uncertainty. If we can't beat this, nothing else matters. Dead simple to implement, sets the floor.

**Rhetorical purpose:** Floor — can you beat the simplest approach?

#### 2. Semantic Uncertainty (Kuhn et al., 2023 / Nature 2024)

The current gold standard for training-free hallucination detection. Clusters multiple generations by semantic equivalence, then computes entropy over clusters. Published in *Nature* — every reviewer will look for this. Our most direct competitor: both are uncertainty-based methods, but ours is cheaper (no multiple generations + semantic clustering needed).

**Paper:** *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation* (ICLR 2023) and *Detecting Hallucinations in Large Language Models Using Semantic Entropy* (Nature, 2024)

**Rhetorical purpose:** Ceiling for training-free — can you match/beat the best without multiple generations?

#### 3. SelfCheckGPT (Manakul et al., EMNLP 2023)

Detects hallucinations by measuring consistency across multiple sampled generations. Training-free and model-agnostic, but expensive (requires multiple generations per query). Important to include because our key advantage is cost: single-pass vs. multiple generations. We need the numbers to prove it.

**Paper:** *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models* (EMNLP 2023)

**Rhetorical purpose:** Cost argument — same detection quality, fraction of the inference cost.

#### 4. LLM-Check (Mitchell et al., 2023)

Uses hidden state covariance structure (SVD-based scores) and attention patterns to estimate confidence. Training-free, single-pass. We're already implementing their hidden state + attention scores — this is the most natural ablation point to show that adding MoE signals on top strictly improves over standard internal signals alone.

**Rhetorical purpose:** Ablation — do MoE signals add value on top of standard internal signals?

### Should Include

#### 5. Semantic Energy (Ma et al., 2025)

Addresses semantic entropy's weakness by operating on penultimate-layer logits with a Boltzmann-inspired energy function instead of post-softmax probabilities. Recent, captures model uncertainty in cases where semantic entropy fails. Related to our internal signal approach — good to show that routing signals capture something logit-space energy doesn't.

**Paper:** *Semantic Energy: Detecting LLM Hallucination Beyond Entropy* (arXiv 2508.14496)

**Rhetorical purpose:** Internal signal comparison — do routing signals beat logit-space energy?

#### 6. HaluNet (Tong et al., 2025)

A trainable multi-branch neural framework that fuses token-level probability uncertainty, semantic embeddings, and distributional uncertainty. Tested on SQuAD, TriviaQA, and Natural Questions (overlapping with our datasets). Good as a trainable upper-bound comparison to show where a trained method sits, even though it's a different category (requires training, not post-hoc).

**Paper:** *HaluNet: Multi-Granular Uncertainty Modeling for Efficient Hallucination Detection in LLM Question Answering* (arXiv 2512.24562)

**Rhetorical purpose:** Trainable upper bound — how close does a training-free approach get?

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
| **Ours** | Internal signal (MoE) | 1 | No (optionally) | All of LLM-Check + routing entropy, expert similarity, expert usage, Gini, Herfindahl |