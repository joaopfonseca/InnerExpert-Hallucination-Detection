# Experiments

This directory contains numbered experiment scripts organized into sections based on the research workflow.

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

### 1.1-analyze-metrics.py

Analyzes generation metrics from step 1.0 to assess their separability between base and RAG outputs. Creates visualizations including:
- Violin plots of metric distributions
- ROC curves for individual metrics and logistic regression
- Confusion matrices at accuracy-optimal thresholds

Used to evaluate whether generation quality metrics can serve as proxies for hallucination detection.

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

- Generates training data for RealtimeQA 2025 (default)
- Labels the data with an LLM-as-judge (`zai-org/GLM-5.1` on DeepInfra by default)
- Trains the MoE detector, fits baseline thresholds, and trains HaluNet
- Saves artefacts to `models/<model_slug>/`:
  - `detector.pkl` – trained MoE detector
  - `thresholds.json` – tuned thresholds for baselines
  - `halunet.pt` – trained HaluNet checkpoint

The training pipeline refuses to overwrite existing artefacts; delete the model directory if you want to re-run.

### 2. Evaluation pipeline

```bash
./experiments/eval_pipeline.sh
```

- Generates OOD test data for RealtimeQA 2026 (default)
- Labels the test data
- Runs every detection method (our detector + all baselines + HaluNet)
- Produces comparison tables, ROC curves, and calibration plots in `data/realtimeqa-2026/<model_slug>/analysis/`

The evaluation pipeline will abort immediately if the training artefacts are missing, and will also abort if predictions already exist for the target test period (to avoid accidental overwrites).

### Configuration

Shared defaults live in `experiments/pipeline_config.sh`:

| Variable | Default | What to change |
|---|---|---|
| `MODEL` | `allenai/OLMoE-1B-7B-0924-Instruct` | Subject model whose outputs are scored |
| `LABEL_MODEL` | `zai-org/GLM-5.1` | LLM used as judge to label hallucinations |
| `TRAIN_YEARS` | `2025` | Year(s) used for training |
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

## Data Structure

```
data/
└── realtimeqa-{year}-{month}/
    └── {model_slug}/
        ├── realtimeqa_original.parquet                # Raw dataset
        ├── results.parquet                            # Generated answers + metrics
        └── results_labeled_{labelling_model}.parquet  # With hallucination labels
```
