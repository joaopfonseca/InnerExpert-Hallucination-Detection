# MoE-Uncertainty-Estimation

Hallucination detection in Mixture-of-Experts (MoE) LLMs via routing-time internal signals.

## Overview

Mixture-of-Experts language models expose internal routing signals that dense models do not provide: which experts a token was routed to, how confidently, and how consistently. This project treats those signals as proxies for epistemic uncertainty and uses them to detect hallucinations at both the answer level and the token level — single-pass, post-hoc, with no modification to the model architecture or training.

The repository contains `moeuncert`, a library for intercepting the MoE forward pass and extracting routing/hidden-state/attention signals, alongside numbered experiment scripts that generate answers, label them, train detectors, and evaluate against a suite of baselines.

## Method

At each generation step, `MoEMonitor` intercepts the forward pass to collect:

- **Standard signals** — hidden state scores, attention scores, top-k output entropy (from LLM-Check)
- **MoE-specific signals** — router entropy, expert hidden scores, expert similarity, expert usage frequency, Gini impurity of expert usage, inverse Herfindahl index

These are combined either with a training-free threshold detector or a trainable classifier (logistic regression / RandomForest / XGBoost / MLP).

## Supported models

| Model | HF id | MoE layout | Default quant |
|---|---|---|---|
| OLMoE-1B-7B-0924-Instruct | `allenai/OLMoE-1B-7B-0924-Instruct` | 64 experts, 8 active | 4-bit |
| Gemma 4 26B A4B IT (text-only) | `google/gemma-4-26B-A4B-it` | 128 + 1 shared, 8 active | 4-bit |

Both are registered in `moeuncert.forwards.MOE_FORWARD_REGISTRY`. To add a new MoE model, implement a `forward_<model>` function in `moeuncert/forwards/_model_forwards.py`, register the block class in `_experts_states.py`, and run the smoke test — see `experiments/README.md` § "Adding a new MoE model" for details.

## Repository layout

```
moeuncert/            # Library: forwards, metrics, monitoring, baselines, models, datasets
experiments/          # Numbered pipeline scripts (0.x–8.x) + shell pipelines
models/               # Trained artefacts (detector.pkl, thresholds.json, halunet.pt)  [gitignored]
data/                 # Generated answers, labels, predictions, analysis  [gitignored]
figures/              # Paper figures  [gitignored]
tables/               # Paper LaTeX tables  [gitignored]
presentation/         # reveal.js talk
pretrained_models/    # HF cache (kept inside the repo)  [gitignored]
Makefile              # Convenience targets (make help)
requirements.txt
```

## Setup

Python ≥ 3.11 is recommended (required by the pinned `torch` / `transformers` / `pandas` versions).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```env
DEEPINFRA_API_KEY=your_deepinfra_api_key_here
```

This key is used by the LLM-as-judge labeler (`zai-org/GLM-5.1` on DeepInfra by default) in `experiments/2.0-make-labels.py` and `experiments/6.1-oos-label.py`. If it is missing, labeling is skipped with a warning.

The `moeuncert` package is imported by the experiment scripts via `sys.path`, so no editable install is required. If you import it from outside `experiments/`, set `PYTHONPATH` to the repository root.

## Run

Three shell pipelines drive the workflow (each is idempotent and skips phases whose outputs already exist):

```bash
# 1. Train: generate RTQA answers, label, train MoE detector + baselines + HaluNet
./experiments/train_pipeline.sh

# 2. Evaluate: OOD test on RTQA 2026, produce comparison tables/plots
./experiments/eval_pipeline.sh

# 3. Cross-dataset OOS: SQuAD, TruthfulQA, NQ-Open, FreshQA
./experiments/oos_pipeline.sh
```

Configuration (model, quantization, label model, train/test years, sampling) lives in `experiments/pipeline_config.sh`. For the per-script reference and CLI flags, see `experiments/README.md`.

## Datasets

- **RealtimeQA** (primary) — temporal split: 2024–2025 for training, 2026 for OOD evaluation.
- **OOS cross-dataset** — SQuAD, TruthfulQA, NQ-Open, FreshQA.

Loaders live in `moeuncert.datasets`; prompt adapters in `moeuncert.datasets_adapters`.

## Documentation

- [`experiments/README.md`](experiments/README.md) — per-script reference and data layout
- [`presentation/README.md`](presentation/README.md) — reveal.js research talk

## Citation

```bibtex
@misc{moe_uncertainty_estimation,
  title  = {Hallucination Detection in Mixture-of-Experts LLMs via Routing-Time Internal Signals},
  author = {<your name>},
  year   = {2026},
  note   = {Preprint in preparation},
  url    = {<repo url>}
}
```

## License

Released under the MIT License.
