"""7.0 — OOS evaluation: run all detection methods on out-of-sample data.

Loads the trained detector, thresholds, and HaluNet from
``models/<model_slug>/`` (trained on RealtimeQA), then runs every
detection method on an OOS dataset's generated answers.  Saves raw
prediction scores to ``data/<dataset_slug>/<model_slug>/predictions/``.

Reuses the shared prediction functions from
``moeuncert.evaluation.predictions`` (also used by 4.0 for the
in-distribution RealtimeQA evaluation).

Outputs:
  - data/<dataset_slug>/<model_slug>/predictions/predictive_entropy*.parquet
  - data/<dataset_slug>/<model_slug>/predictions/llm_check.parquet
  - data/<dataset_slug>/<model_slug>/predictions/detector_*.parquet
  - data/<dataset_slug>/<model_slug>/predictions/halunet.parquet
  - data/<dataset_slug>/<model_slug>/predictions/semantic_uncertainty.parquet
  - data/<dataset_slug>/<model_slug>/predictions/semantic_energy.parquet
  - data/<dataset_slug>/<model_slug>/predictions/selfcheck_*.parquet
  - data/<dataset_slug>/<model_slug>/predictions/ground_truth.parquet

Usage:
    python 7.0-oos-evaluation.py --dataset squad
    python 7.0-oos-evaluation.py --dataset truthfulqa --skip-sampled
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.datasets_adapters import get_adapter, list_oos_datasets
from moeuncert.experiments import (
    resolve_model_slug,
    resolve_cache_dir,
)
from moeuncert.experiments.data_loading import load_model_outputs, load_labeled_dataset
from moeuncert.evaluation import (
    build_ground_truth,
    build_composite_qids,
    build_label_lookup,
    evaluate_detector,
    evaluate_halunet,
    evaluate_individual_signal,
    evaluate_llm_check,
    evaluate_predictive_entropy,
    evaluate_selfcheck,
    evaluate_semantic_energy,
    evaluate_semantic_uncertainty,
)


# ---------------------------------------------------------------------------
# OOS sampled-outputs loader (mirrors load_sampled_outputs but without
# year/month resolution — OOS datasets use a flat oos-<name> slug).
# ---------------------------------------------------------------------------


def load_oos_sampled_outputs(data_dir: Path, num_samples: int = 5, subdir: str = "sampled_generation") -> dict:
    """Load sampled outputs from an OOS dataset directory.

    ``data_dir`` is ``data/<dataset_slug>/<model_slug>/`` containing a
    ``sampled_generation/`` (or ``sampled_generation_evidence/``) subdirectory.
    """
    from moeuncert.experiments.utils import read_and_collate_outputs
    from moeuncert.experiments.data_loading import _normalize_question_id_value

    sampled_dir = data_dir / subdir
    if not sampled_dir.exists():
        return {"responses_by_qid": {}, "log_probs_by_qid": {}, "logits_by_qid": {}}

    batch_files = sorted(sampled_dir.glob("sampled_outputs__batch_*.pt"))
    if not batch_files:
        return {"responses_by_qid": {}, "log_probs_by_qid": {}, "logits_by_qid": {}}

    outputs = read_and_collate_outputs(batch_files, tokenizer=None, get_keys=None)
    print(f"  Loaded {len(batch_files)} sampled batch files")

    all_responses_by_qid = {}
    all_logprobs_by_qid = {}
    all_logits_by_qid = {}

    qids = outputs["question_id"]
    for idx, qid in enumerate(qids):
        qid = _normalize_question_id_value(qid)
        responses = []
        log_probs = []
        logits = []

        for s in range(num_samples):
            seq_key = f"sequences_sample{s}"
            ll_key = f"log_likelihoods_sample{s}"
            scores_key = f"scores_sample{s}"

            if seq_key not in outputs or ll_key not in outputs:
                continue

            gen_tokens = outputs[seq_key][idx]
            gen_tokens = gen_tokens[gen_tokens != 0]
            responses.append(gen_tokens.tolist())
            log_probs.append(outputs[ll_key][idx].tolist())

            if scores_key in outputs:
                scores = outputs[scores_key][idx]
                gen_len = min(len(outputs[ll_key][idx]), scores.shape[0])
                if gen_len > 0 and len(gen_tokens) > 0:
                    token_ids = gen_tokens[:gen_len]
                    response_logits = scores[:gen_len].gather(
                        -1, token_ids.unsqueeze(-1).to(scores.device)
                    ).squeeze(-1).tolist()
                    logits.append(response_logits)

        if responses:
            all_responses_by_qid[qid] = responses
            all_logprobs_by_qid[qid] = log_probs
            if logits:
                all_logits_by_qid[qid] = logits

    return {
        "responses_by_qid": all_responses_by_qid,
        "log_probs_by_qid": all_logprobs_by_qid,
        "logits_by_qid": all_logits_by_qid,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run all detection methods on OOS data (7.0)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list_oos_datasets(),
        help="OOS dataset name",
    )
    parser.add_argument(
        "--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--label-model", type=str, default="zai-org/GLM-5.1",
        help="Label model name (for finding the labeled parquet)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="Directory with saved trained models",
    )
    parser.add_argument(
        "--skip-sampled", action="store_true",
        help="Skip methods requiring multi-sample data (SU, SE, SelfCheck)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of sampled responses per question (default: 5)",
    )

    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"7.0 — OOS EVALUATION ({args.dataset})")
    print(f"{'=' * 70}")

    adapter = get_adapter(args.dataset)
    model_slug = resolve_model_slug(args.model)
    model_dir = args.data_root / adapter.slug / model_slug

    if not model_dir.exists():
        raise FileNotFoundError(
            f"OOS data directory not found: {model_dir}\n"
            f"Run 6.0-oos-generate.py --dataset {args.dataset} first."
        )

    # --- Load data --------------------------------------------------------
    print("\nLoading OOS data...")
    from moeuncert.experiments import load_tokenizer_for_data
    tokenizer = load_tokenizer_for_data(args.model)

    df_labeled = load_labeled_dataset(model_dir, label_model=args.label_model)
    print(f"  {len(df_labeled)} labeled rows")

    outputs = load_model_outputs(model_dir, pad_token_id=tokenizer.pad_token_id)
    print(f"  Outputs loaded: {len(outputs.get('question_id', []))} samples")

    comp_qids = build_composite_qids(outputs)
    label_lookup = build_label_lookup(df_labeled)

    # --- Predictions dir --------------------------------------------------
    predictions_dir = model_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # --- PredictiveEntropy ------------------------------------------------
    print("\n[1/7] PredictiveEntropy ...")
    pe_results = evaluate_predictive_entropy(outputs, comp_qids, label_lookup)
    for name, df in pe_results.items():
        path = predictions_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")

    # --- LLM-Check --------------------------------------------------------
    print("\n[2/7] LLM-Check ...")
    thresholds_path = args.models_dir / model_slug / "thresholds.json"
    llm_thresholds = None
    if thresholds_path.exists():
        with open(thresholds_path, "r") as f:
            llm_thresholds = json.load(f)
        print(f"  Loaded thresholds from {thresholds_path}")
    else:
        print(f"  WARNING: {thresholds_path} not found — LLM-Check will use all-layer average (legacy).")
    try:
        llm_df = evaluate_llm_check(outputs, comp_qids, label_lookup, thresholds=llm_thresholds)
        path = predictions_dir / "llm_check.parquet"
        llm_df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(llm_df)} rows)")
    except KeyError as e:
        print(f"  SKIPPED — {e}")

    # --- Individual MoE Signals ------------------------------------------
    print("\n[2b/7] Individual MoE Signals ...")
    INDIVIDUAL_SIGNALS = [
        "router_entropy",
        "expert_hidden_scores",
        "expert_similarities",
        "expert_usage_entropy",
        "expert_usage_gini",
        "expert_usage_effective_experts",
    ]
    for signal_name in INDIVIDUAL_SIGNALS:
        if signal_name not in outputs:
            print(f"  SKIPPED — {signal_name} not in outputs")
            continue
        try:
            sig_df = evaluate_individual_signal(
                outputs, comp_qids, signal_name, thresholds=llm_thresholds
            )
            path = predictions_dir / f"signal_{signal_name}.parquet"
            sig_df.to_parquet(path, index=False)
            print(f"  Saved {path} ({len(sig_df)} rows)")
        except Exception as e:
            print(f"  SKIPPED {signal_name} — {e}")

    # --- MoE Detector ----------------------------------------------------
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

    best_detector_path = models_subdir / "detector.pkl"
    if best_detector_path.exists():
        det_df = evaluate_detector(outputs, df_labeled, best_detector_path)
        path = predictions_dir / "detector.parquet"
        det_df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(det_df)} rows) [best-family alias]")

    # --- HaluNet ----------------------------------------------------------
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

    # --- Sampled-data methods ---------------------------------------------
    if args.skip_sampled:
        print("\n[5-7] Skipping sampled-data methods (--skip-sampled)")
    else:
        print("\n[5-7] Loading sampled data...")
        sampled = load_oos_sampled_outputs(model_dir, num_samples=args.num_samples)
        n_qids = len(sampled.get("responses_by_qid", {}))
        print(f"  {n_qids} questions with sampled responses (base)")

        # Load evidence sampled data if available
        sampled_ev = None
        sampled_ev_dir = model_dir / "sampled_generation_evidence"
        if sampled_ev_dir.exists():
            sampled_ev = load_oos_sampled_outputs(model_dir, num_samples=args.num_samples, subdir="sampled_generation_evidence")
            n_qids_ev = len(sampled_ev.get("responses_by_qid", {}))
            print(f"  {n_qids_ev} questions with sampled responses (evidence)")

        if n_qids == 0 and (sampled_ev is None or len(sampled_ev.get("responses_by_qid", {})) == 0):
            print("  No sampled data found — skipping SU/SE/SelfCheck")
        else:
            for ev_present, sampled_data, label in [
                (0, sampled, "base"),
                (1, sampled_ev, "evidence"),
            ]:
                if sampled_data is None or len(sampled_data.get("responses_by_qid", {})) == 0:
                    continue
                print(f"\n  [{label} condition]")

                # SemanticUncertainty
                print(f"\n[5/7] SemanticUncertainty ({label}) ...")
                try:
                    su_df = evaluate_semantic_uncertainty(
                        args.model, sampled=sampled_data, tokenizer=tokenizer,
                        evidence_present=ev_present,
                    )
                    path = predictions_dir / "semantic_uncertainty.parquet"
                    if path.exists() and ev_present == 1:
                        existing = pd.read_parquet(path)
                        su_df = pd.concat([existing, su_df], ignore_index=True)
                    su_df.to_parquet(path, index=False)
                    print(f"  Saved {path} ({len(su_df)} rows)")
                except Exception as e:
                    print(f"  SKIPPED — error: {e}")

                # SemanticEnergy
                print(f"\n[6/7] SemanticEnergy ({label}) ...")
                try:
                    se_df = evaluate_semantic_energy(
                        args.model, sampled=sampled_data, tokenizer=tokenizer,
                        evidence_present=ev_present,
                    )
                    path = predictions_dir / "semantic_energy.parquet"
                    if path.exists() and ev_present == 1:
                        existing = pd.read_parquet(path)
                        se_df = pd.concat([existing, se_df], ignore_index=True)
                    se_df.to_parquet(path, index=False)
                    print(f"  Saved {path} ({len(se_df)} rows)")
                except Exception as e:
                    print(f"  SKIPPED — error: {e}")

                # SelfCheckGPT
                print(f"\n[7/7] SelfCheckGPT ({label}) ...")
                for variant in ["nli", "prompt"]:
                    try:
                        sc_df = evaluate_selfcheck(
                            args.model, variant=variant, sampled=sampled_data,
                            tokenizer=tokenizer, evidence_present=ev_present,
                        )
                        path = predictions_dir / f"selfcheck_{variant}.parquet"
                        if path.exists() and ev_present == 1:
                            existing = pd.read_parquet(path)
                            sc_df = pd.concat([existing, sc_df], ignore_index=True)
                        sc_df.to_parquet(path, index=False)
                        print(f"  Saved {path} ({len(sc_df)} rows)")
                    except Exception as e:
                        print(f"  selfcheck_{variant} SKIPPED — error: {e}")

    # --- Ground Truth -----------------------------------------------------
    print("\n[GT] Building ground truth ...")
    gt_df = build_ground_truth(outputs, df_labeled, label_lookup, tokenizer)
    gt_path = predictions_dir / "ground_truth.parquet"
    gt_df.to_parquet(gt_path, index=False)
    print(f"  Saved {gt_path} ({len(gt_df)} rows)")

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Predictions saved to: {predictions_dir}")
    print(f"Next step: python 8.0-oos-analysis.py --datasets {args.dataset}")


if __name__ == "__main__":
    main()