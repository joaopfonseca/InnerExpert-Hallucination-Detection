"""Run all hallucination detection methods on OOD test data and save predictions.

This script is the HEADLINE EVALUATION of the project. It runs inference
with every method (our trained detector, all baselines, HaluNet) on the
held-out 2026 RealtimeQA test data and saves raw prediction scores.

Methods evaluated:
  - PredictiveEntropy (per-token + answer-level via mean/max aggregation)
  - LLM-Check (attention, hidden, perplexity, entropy — per-token + answer-level)
  - SemanticUncertainty (answer-level, requires multi-sample data)
  - SemanticEnergy (answer-level, requires multi-sample data)
  - SelfCheckGPT (NLI + Prompt — answer-level, requires multi-sample data)
  - HaluNet (answer-level, requires log_likelihoods/entropies/embeddings)
  - Our MoE detector (per-token, from detector.pkl)

Saved models (from 3.X scripts):
  - models/<model_slug>/detector.pkl
  - models/<model_slug>/halunet.pt
  - models/<model_slug>/thresholds.json

This script requires model outputs saved by 1.0-generate-answers.py.

**Dependencies on batch file keys:**
  - PredictiveEntropy: requires `scores` (raw output logits, shape B×seq×vocab).
    If unavailable, falls back to `scores_entropy` (top-k entropy, less ideal).
  - LLM-Check: uses pre-computed `attention_scores`, `hidden_scores`,
    and `scores_entropy` from batch files. `sequences` + `input_ids` for perplexity.
  - HaluNet: requires `log_likelihoods` and `entropies` (return_baseline_features=True).
  - SU/SE/SelfCheckGPT: require multi-sample data via load_sampled_outputs.

Output:
  - data/<dataset_slug>/<model_slug>/predictions/*.parquet (one per method)
  - data/<dataset_slug>/<model_slug>/predictions/ground_truth.parquet

Usage:
    python 4.0-model-evaluation.py --test-years 2026 --test-month 1
    python 4.0-model-evaluation.py --test-years 2026 --test-month 2 --model allenai/OLMoE-1B-7B-0924-Instruct
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))

from moeuncert.experiments import (
    resolve_model_slug,
    resolve_dataset_slug,
    resolve_cache_dir,
    load_multi_year_data,
)

# Backward-compat alias for pickles trained before _replace_inf_with_nan
# moved to moeuncert.experiments.utils.
from moeuncert.experiments import replace_inf_with_nan
_replace_inf_with_nan = replace_inf_with_nan

from moeuncert.evaluation import (
    build_ground_truth,
    evaluate_detector,
    evaluate_halunet,
    evaluate_llm_check,
    evaluate_predictive_entropy,
    evaluate_selfcheck,
    evaluate_semantic_energy,
    evaluate_semantic_uncertainty,
)


# ---------------------------------------------------------------------------
# Helpers (imported from moeuncert.evaluation)
# ---------------------------------------------------------------------------

# Re-export the shared helpers under their old private names so the
# existing main() body can reference them without churn.
from moeuncert.evaluation.predictions import (
    build_composite_qids as _build_composite_qids,
    build_label_lookup as _build_label_lookup,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all hallucination detection methods on OOD test data"
    )
    parser.add_argument(
        "--test-years", type=int, nargs="+", default=[2026],
        help="Test years (default: 2026)",
    )
    parser.add_argument(
        "--test-month", type=int, default=None,
        help="Test month (for single-year test datasets)",
    )
    parser.add_argument(
        "--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--label-model", type=str, default="zai-org/GLM-5.1",
        help="Label model name",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="Directory with saved models",
    )
    parser.add_argument(
        "--skip-sampled", action="store_true",
        help="Skip methods that require multi-sample data (SU, SE, SelfCheckGPT)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sampled responses per question for SU/SE/SelfCheck (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used during generation (default: 0.7)",
    )

    args = parser.parse_args()

    print(f"{'=' * 70}")
    print("4.0 — MODEL EVALUATION (OOD INFERENCE)")
    print(f"{'=' * 70}")

    model_slug = resolve_model_slug(args.model)

    # Load data first; determine the actual source directory afterwards.
    print("\nLoading test data...")
    from moeuncert.experiments import load_tokenizer_for_data
    tokenizer = load_tokenizer_for_data(args.model)
    df_labeled, outputs, source_dir = load_multi_year_data(
        args.data_root, args.test_years, args.test_month,
        args.model, args.label_model,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"  {len(df_labeled)} labeled rows")
    print(f"  Source data dir: {source_dir}")

    comp_qids = _build_composite_qids(outputs)
    label_lookup = _build_label_lookup(df_labeled)

    # Create predictions dir inside the actual data source, never in a phantom path.
    predictions_dir = source_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Test dataset: {args.test_years}")
    if args.test_month:
        print(f"Test month: {args.test_month:02d}")

    # -----------------------------------------------------------------------
    # PredictiveEntropy
    # -----------------------------------------------------------------------
    print("\n[1/7] PredictiveEntropy ...")
    pe_results = evaluate_predictive_entropy(outputs, comp_qids, label_lookup)
    for name, df in pe_results.items():
        path = predictions_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # LLM-Check
    # -----------------------------------------------------------------------
    print("\n[2/7] LLM-Check ...")
    llm_df = evaluate_llm_check(outputs, comp_qids, label_lookup)
    path = predictions_dir / "llm_check.parquet"
    llm_df.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(llm_df)} rows)")

    # -----------------------------------------------------------------------
    # Our Detector (every candidate family from 3.0)
    # -----------------------------------------------------------------------
    print("\n[3/7] MoE Detector ...")
    models_subdir = args.models_dir / model_slug
    detector_pickles = sorted(models_subdir.glob("detector_*.pkl"))
    if not detector_pickles:
        print(f"  SKIPPED — no detector_*.pkl found in {models_subdir}")
    else:
        for detector_path in detector_pickles:
            family = detector_path.stem[len("detector_"):]
            det_df = evaluate_detector(outputs, df_labeled, detector_path)
            path = predictions_dir / f"detector_{family}.parquet"
            det_df.to_parquet(path, index=False)
            print(f"  Saved {path} ({len(det_df)} rows)")

    # Backward-compat: also emit the canonical detector.parquet from the
    # overall-best detector.pkl so any existing consumer keeps working.
    best_detector_path = models_subdir / "detector.pkl"
    if best_detector_path.exists():
        det_df = evaluate_detector(outputs, df_labeled, best_detector_path)
        path = predictions_dir / "detector.parquet"
        det_df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(det_df)} rows) [best-family alias]")

    # -----------------------------------------------------------------------
    # HaluNet
    # -----------------------------------------------------------------------
    print("\n[4/7] HaluNet ...")
    halunet_path = args.models_dir / model_slug / "halunet.pt"
    if halunet_path.exists() and "log_likelihoods" in outputs:
        try:
            hnet_df = evaluate_halunet(outputs, df_labeled, halunet_path)
            path = predictions_dir / "halunet.parquet"
            hnet_df.to_parquet(path, index=False)
            print(f"  Saved {path} ({len(hnet_df)} rows)")
        except Exception as e:
            print(f"  SKIPPED — error: {e}")
    else:
        missing = []
        if not halunet_path.exists():
            missing.append(f"halunet.pt at {halunet_path}")
        if "log_likelihoods" not in outputs:
            missing.append("log_likelihoods in outputs")
        print(f"  SKIPPED — missing: {', '.join(missing)}")

    # -----------------------------------------------------------------------
    # Sampled-data methods
    # -----------------------------------------------------------------------
    if args.skip_sampled:
        print("\n[5-7] Skipping sampled-data methods (-skip-sampled)")
    else:
        # SemanticUncertainty
        print("\n[5/7] SemanticUncertainty ...")
        try:
            su_df = evaluate_semantic_uncertainty(
                args.model, args.test_years, args.test_month,
                args.data_root, num_samples=args.num_samples,
            )
            path = predictions_dir / "semantic_uncertainty.parquet"
            su_df.to_parquet(path, index=False)
            print(f"  Saved {path} ({len(su_df)} rows)")
        except Exception as e:
            print(f"  SKIPPED - error: {e}")

        # SemanticEnergy
        print("\n[6/7] SemanticEnergy ...")
        try:
            se_df = evaluate_semantic_energy(
                args.model, args.test_years, args.test_month,
                args.data_root, num_samples=args.num_samples,
            )
            path = predictions_dir / "semantic_energy.parquet"
            se_df.to_parquet(path, index=False)
            print(f"  Saved {path} ({len(se_df)} rows)")
        except Exception as e:
            print(f"  SKIPPED - error: {e}")

        # SelfCheckGPT (NLI + Prompt)
        print("\n[7/7] SelfCheckGPT ...")
        for variant in ["nli", "prompt"]:
            try:
                sc_df = evaluate_selfcheck(
                    args.model, args.test_years, args.test_month,
                    args.data_root, variant=variant,
                    num_samples=args.num_samples,
                )
                path = predictions_dir / f"selfcheck_{variant}.parquet"
                sc_df.to_parquet(path, index=False)
                print(f"  Saved {path} ({len(sc_df)} rows)")
            except Exception as e:
                print(f"  selfcheck_{variant} SKIPPED - error: {e}")

    # -----------------------------------------------------------------------
    # Ground Truth
    # -----------------------------------------------------------------------
    print("\n[GT] Building ground truth ...")
    gt_df = build_ground_truth(outputs, df_labeled, label_lookup, tokenizer)
    gt_path = predictions_dir / "ground_truth.parquet"
    gt_df.to_parquet(gt_path, index=False)
    print(f"  Saved {gt_path} ({len(gt_df)} rows)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Predictions saved to: {predictions_dir}")
    print(f"Files: {[f.name for f in sorted(predictions_dir.glob('*.parquet'))]}")
    print("\nNext step: python 5.0-results-analysis.py")


if __name__ == "__main__":
    main()
