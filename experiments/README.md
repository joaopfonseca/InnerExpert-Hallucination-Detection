# Experiments

This directory contains numbered experiment scripts organized into sections based on the research workflow.

## Supported Models

Both models below are registered in `moeuncert.forwards.MOE_FORWARD_REGISTRY`,
so the MoE-instrumentation (router logits, expert hidden states, expert usage
patterns, etc.) works for either one out of the box. To switch models, change
`$MODEL` in `pipeline_config.sh` (or override at the command line) and re-run
the pipeline.

| Model | HF id | MoE layout | Default quant | Slug |
|---|---|---|---|---|
| OLMoE-1B-7B-0924-Instruct | `allenai/OLMoE-1B-7B-0924-Instruct` | 64 experts, 8 active | 4-bit | `allenai__OLMoE-1B-7B-0924-Instruct` |
| Gemma 4 26B A4B IT (text-only) | `google/gemma-4-26B-A4B-it` | 128 experts + 1 shared, 8 active | 4-bit | `google__gemma-4-26B-A4B-it` |

Gemma 4 is a multimodal model (text + image + audio for the smaller variants);
we use it as a text-only `AutoModelForCausalLM` in this pipeline. Its MoE
block class is `Gemma4TextExperts` and its host layer
(`Gemma4TextDecoderLayer`) flattens hidden states before calling the experts,
which our registry handles automatically.

### Smoke testing a new model

Before running the full pipeline on a new model, run the smoke test to verify
the MoE instrumentation produces the expected tensor shapes:

```bash
python experiments/_smoke_test_moe.py --model google/gemma-4-26B-A4B-it
```

The script loads the model, runs a 4-token `generate()`, and asserts that
`experts_hidden` is non-empty and has the expected 4D layout
`(batch, seq, top_k, hidden)`. Exits 0 on success.

### Adding a new MoE model

1. Implement a new `forward_<model>` function in
   `moeuncert/forwards/_model_forwards.py`. The forward must set
   `self.last_experts_hidden` to a dict with keys `expert_idx`,
   `expert_weights`, `expert_hidden_states` matching the layout used by
   `forward_olmoe` (see `moeuncert/metrics/_metrics.py` for shapes).
2. Import the MoE block class in
   `moeuncert/forwards/_experts_states.py` and add an entry to
   `MOE_FORWARD_REGISTRY`. If the new block's host layer flattens hidden
   states before calling the experts, also wrap the import in a
   `try/except` and use `_install_parent_shape_capture` (the same flow used
   for Gemma 4) so the forward can recover batch/seq.
3. Run the smoke test to confirm the registry picks it up.

## Section 0: Exploration & Testing

### 0.0-hallucination-detection-test.ipynb

Exploratory notebook investigating whether internal model signals (attention scores, hidden state scores) can distinguish between hallucinated and factual content. Tests with GPT-2 and SmolLM2 models on synthetic prompts, comparing statistical distribution differences using Wasserstein distance, energy distance, KS tests, and MMD.

### 0.1-moe-uncertainty-exploratory.py

Playground script for testing Mixture-of-Experts signal collection and model output reconstruction with OLMoE-1B-7B. Validates that autoregressive generation outputs can be faithfully reconstructed with full hidden states, router logits, and expert activations.

### 0.2-compute-monitor-metrics.py

Computes various MoE monitoring metrics on RealtimeQA samples with and without evidence. Metrics include:
- Hidden state scores
- Attention scores
- Top-k entropy
- Router entropy
- Expert hidden scores
- Expert similarity scores
- Expert usage patterns

Outputs visualizations of router entropy and expert usage across layers.

---

## Section 1: Answer Generation & Analysis

### 1.0-generate-answers.py

Generates answers to RealtimeQA questions using an instruction-tuned LLM. Produces two outputs:
- **Base generation**: Answers without evidence
- **Evidence-based generation (RAG)**: Answers with retrieved evidence

Computes evaluation metrics (ROUGE, BERTScore, BLEU) comparing generated answers against references. Saves results to `data/<dataset_slug>/<model_slug>/results.parquet`.

### 1.1-generate-baseline-samples.py

Generates multiple stochastic samples per question for sampling-based hallucination
baselines (Semantic Uncertainty, Semantic Energy, SelfCheckGPT). Saves them to
`data/<dataset_slug>/<model_slug>/sampled_generation/`.

---

## Section 2: Label Creation

### 2.0-make-labels.py

Creates answer-level and token-level hallucination labels for RealtimeQA outputs. Combines:

- **Weak labels**: Evidence presence + metric thresholds (ROUGE, BERTScore, BLEU)
- **LLM labels** (optional): Ollama-based classification with hallucinated span extraction
- **Logistic regression classifier**: Trained on LLM labels when available

Outputs a labeled parquet file with columns:
- `label_hallucination_confidence`: Continuous confidence score
- `label_hallucinated_answer`: Binary answer-level label
- `label_token_hallucination_mask`: Token-level hallucination mask
- `label_token_hallucination_ratio`: Proportion of hallucinated tokens
- `label_answer_source` / `label_token_source`: Source of each label

---

## Full Pipeline End-to-End

The easiest way to run the full workflow is via the two shell pipelines in this directory.

### 1. Training pipeline

```bash
./experiments/train_pipeline.sh
```

- Generates training data for RealtimeQA 2024-2025 (default)
- Labels the data with an LLM-as-judge (`zai-org/GLM-5.1` on DeepInfra by default)
- Trains the MoE detector, fits baseline thresholds, and trains HaluNet
- Saves artefacts to `models/<model_slug>/`:
  - `detector.pkl` – trained MoE detector
  - `thresholds.json` – tuned thresholds for baselines
  - `halunet.pt` – trained HaluNet checkpoint

The training pipeline skips phases that already have the expected outputs (no
overwrites) — delete the model directory if you want to re-run from scratch.

### 2. Evaluation pipeline

```bash
./experiments/eval_pipeline.sh
```

- Generates OOD test data for RealtimeQA 2026 (default)
- Labels the test data
- Runs every detection method (our detector + all baselines + HaluNet)
- Produces comparison tables, ROC curves, and calibration plots in `data/realtimeqa-2026/<model_slug>/analysis/`

The evaluation pipeline will abort immediately if the training artefacts are
missing. Phases that already have outputs (results, labels, predictions,
analysis) are skipped automatically to avoid redundant work.

### Configuration

Shared defaults live in `experiments/pipeline_config.sh`:

| Variable | Default | What to change |
|---|---|---|
| `MODEL` | `allenai/OLMoE-1B-7B-0924-Instruct` | Subject model whose outputs are scored (see [Supported Models](#supported-models)) |
| `QUANTIZE` | `4-bit` | One of `16-bit`, `8-bit`, `4-bit` |
| `LABEL_MODEL` | `zai-org/GLM-5.1` | LLM used as judge to label hallucinations |
| `TRAIN_YEARS` | `2024 2025` | Year(s) used for training |
| `TRAIN_MONTH` | *(empty)* | Leave empty for all months; set to e.g. `1` for January only |
| `TEST_YEARS` | `2026` | Year(s) used for OOD evaluation |
| `TEST_MONTH` | *(empty)* | Leave empty for all months; set to e.g. `1` for January only |
| `NUM_SAMPLES` | `5` | Number of stochastic samples per question for sampling-based baselines |

### Manual / ad-hoc workflow

If you prefer to run steps individually (e.g. with a custom year or month), invoke the numbered Python scripts directly:

```bash
# 1. Generate answers with evidence comparison
python experiments/1.0-generate-answers.py --model allenai/OLMoE-1B-7B-0924-Instruct --years 2025 --month 1

# 2. Create hallucination labels
python experiments/2.0-make-labels.py --model allenai/OLMoE-1B-7B-0924-Instruct --years 2025 --month 1 --deepinfra-model zai-org/GLM-5.1
```

### Model cache (`pretrained_models/`)

All HuggingFace downloads (weights, tokenizers, configs) are cached **inside**
the project at `pretrained_models/` instead of the global
`~/.cache/huggingface/`. This keeps the repo self-contained and avoids filling
up the user's home directory.

The shell pipelines set `HF_HOME` automatically via `pipeline_config.sh`. When
running Python scripts directly, the code still passes `cache_dir` to every
`from_pretrained()` call, so the local cache is used even if the environment
variable is not set.

Per-model subdirectories follow HuggingFace's naming convention:
```
pretrained_models/
└── models--<org>--<name>/
    ├── snapshots/
    └── ...
```

If you already have the model cached globally and want to avoid re-downloading,
you have three options:

1. **Symlink** the global cache entry into `pretrained_models/`:
   ```bash
   ln -s ~/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924-Instruct \
          pretrained_models/models--allenai--OLMoE-1B-7B-0924-Instruct
   ```
2. **Override** `HF_HOME` in `pipeline_config.sh` (or your shell) to point back
   to the global cache:
   ```bash
   export HF_HOME="$HOME/.cache/huggingface"
   ```
3. **Set** `--cache-dir` manually when running individual scripts:
   ```bash
   python experiments/1.0-generate-answers.py --model ... --cache-dir /mnt/bigdisk/hf_cache
   ```

## Data Structure

```
data/
└── realtimeqa-{year1}[-{year2}[-...]][-{MM}]/
    ├── realtimeqa_original.parquet                # Raw dataset (shared across models)
    └── {model_slug}/
        ├── results.parquet                            # Generated answers + metrics
        ├── results_labeled_{labelling_model}.parquet  # With hallucination labels
        ├── base_generation/                # 1.0 output tensors (no evidence)
        │   └── model_outputs__batch_*.pt
        ├── evidence_generation/            # 1.0 output tensors (with evidence)
        │   └── model_outputs__batch_*.pt
        ├── sampled_generation/             # 1.1 sampled responses
        │   └── sampled_outputs__batch_*.pt
        ├── predictions/                    # 4.0 per-method predictions
        │   ├── predictive_entropy.parquet
        │   ├── llm_check.parquet
        │   ├── detector.parquet
        │   ├── halunet.parquet
        │   ├── semantic_uncertainty.parquet
        │   ├── semantic_energy.parquet
        │   ├── selfcheck_nli.parquet
        │   ├── selfcheck_prompt.parquet
        │   └── ground_truth.parquet
        └── analysis/                       # 5.0 results, tables, plots
            ├── comparison_table.md
            ├── comparison_table_answer.csv
            ├── comparison_table_token.csv
            └── *.png

models/
└── {model_slug}/
    ├── detector.pkl                    # Trained MoE detector (3.0)
    ├── thresholds.json                 # Tuned baseline thresholds (3.1)
    ├── halunet.pt                      # HaluNet checkpoint (3.2)
    ├── halunet_train_summary.json
    ├── train_summary.json
    └── val_results.json
```
