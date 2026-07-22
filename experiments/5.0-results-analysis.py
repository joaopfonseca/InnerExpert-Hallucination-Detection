"""
5.0-results-analysis.py — Analyse per-answer and per-token prediction files.

Reads all *.parquet files under ``--predictions-dir``, aligns them with
``ground_truth.parquet``, and computes:

* Answer-level: AUROC, AUPRC, F1, Accuracy, TPR@1/5/10%FPR, ECE
* Token-level : AUROC, AUPRC, F1, Accuracy, TPR@1/5/10%FPR, ECE

Produces:
  * comparison_table.md          — Markdown tables (answer + token level)
  * comparison_table_answer.csv  — Answer-level metrics, one row per method
  * comparison_table_token.csv   — Token-level metrics, one row per method
  * roc_curves.png               — Overlaid ROC curves
  * calibration_plots.png        — Reliability diagrams
  * results.json                 — Raw numbers for the paper

Usage
-----
    python experiments/5.0-results-analysis.py \\
        --predictions-dir data/realtimeqa-2025-2026/allenai__OLMoE-1B-7B-0924-Instruct/predictions/predictions \\
        [--thresholds-file models/allenai__OLMoE-1B-7B-0924-Instruct/thresholds.json]

If ``--thresholds-file`` is provided, F1 / Accuracy / TPR@X%FPR / ECE are
computed at the tuned threshold for each baseline.  Threshold-independent
metrics (AUROC, AUPRC) never change.  Missing or invalid thresholds fall
back to 0.5.

The script is defensive: missing or empty methods are skipped rather than
crashing, so it can be run incrementally as different parts of the
pipeline finish.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

sys.path.append(str(Path(__file__).parent.parent))

from moeuncert.experiments.paths import resolve_model_slug, resolve_dataset_slug
from moeuncert.evaluation.metrics import compute_ece, compute_metrics_for_predictions

from moeuncert.experiments.data_loading import _find_combined_dataset_dir, _parse_dataset_dir_name

# Backward-compat alias so code in this file can still call _compute_metrics.
_compute_metrics = compute_metrics_for_predictions


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _is_bad_qid(series: pd.Series) -> bool:
    """Detect tensor-repr leftovers, e.g. 'tensor(202501030)'."""
    if series.empty or series.dtype != object:
        return False
    return series.astype(str).str.startswith("tensor(").any()


def _load_prediction_df(
    predictions_dir: Path,
    filename: str,
    required_cols: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    path = predictions_dir / filename
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if len(df) == 0:
        return None
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None
    if _is_bad_qid(df.get("question_id", pd.Series(dtype=object))):
        return None
    return df


def _build_answer_labels(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ground-truth to one answer-label per question_id."""
    return (
        gt_df[["question_id", "answer_label"]]
        .groupby("question_id", as_index=False)
        .first()
        .rename(columns={"answer_label": "label"})
    )


def _build_token_labels(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Token-level labels, keeping token_position."""
    return gt_df[["question_id", "token_position", "token_label"]].rename(
        columns={"token_label": "label"}
    )


# ---------------------------------------------------------------------------
# Method evaluation helpers
# ---------------------------------------------------------------------------

MethodInfo = Tuple[str, Dict[str, float]]


def _get_threshold(thresholds: Optional[Dict[str, Dict]], key: str) -> float:
    """Safely extract a threshold value from the thresholds dict.

    Falls back to 0.5 if the key is missing, the entry has no
    'threshold' field, or the thresholds dict itself is None.
    """
    if thresholds is None:
        return 0.5
    entry = thresholds.get(key, {})
    if not isinstance(entry, dict):
        return 0.5
    t = entry.get("threshold")
    if t is None:
        return 0.5
    try:
        return float(t)
    except (TypeError, ValueError):
        return 0.5


def _eval_answer_level(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate an answer-level prediction DataFrame."""
    merged = gt_df[["question_id", label_col]].merge(
        pred_df[["question_id", score_col]],
        on="question_id",
        how="inner",
    )
    if len(merged) == 0:
        return _compute_metrics(np.array([]), np.array([]))
    y_true  = merged[label_col].astype(int).values
    y_proba = merged[score_col].astype(float).values
    return _compute_metrics(y_true, y_proba, threshold)


def _eval_token_level(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    score_col: str,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a token-level prediction DataFrame."""
    merged = gt_df[["question_id", "token_position", "label"]].merge(
        pred_df[["question_id", "token_position", score_col]],
        on=["question_id", "token_position"],
        how="inner",
    )
    if len(merged) == 0:
        return _compute_metrics(np.array([]), np.array([]))
    y_true  = merged["label"].astype(int).values
    y_proba = merged[score_col].astype(float).values
    return _compute_metrics(y_true, y_proba, threshold)


def _aggregate_token_to_answer(
    token_df: pd.DataFrame,
    score_col: str,
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate per-token scores to one per question_id."""
    grouped = token_df.groupby("question_id", as_index=False)
    if agg == "mean":
        return grouped[score_col].mean()
    elif agg == "max":
        return grouped[score_col].max()
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


def _ours_method_to_pred_fn(method: str) -> Optional[str]:
    """Map an ``Ours-<Family> (mean|max)`` answer-level method name to its
    ``detector_<Family>.parquet`` prediction file.

    Returns ``None`` if *method* is not an ``Ours-`` method.  The family name
    is everything between the ``Ours-`` prefix and the `` (mean)`` / `` (max)``
    aggregation suffix.
    """
    if not method.startswith("Ours-"):
        return None
    family = method[len("Ours-"):]
    for suffix in (" (mean)", " (max)"):
        if family.endswith(suffix):
            family = family[: -len(suffix)]
            break
    return f"detector_{family}.parquet"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(
    predictions_dir: Path,
    output_dir: Path,
    plot_dpi: int = 300,
    thresholds: Optional[Dict[str, Dict]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------------
    gt_df = _load_prediction_df(predictions_dir, "ground_truth.parquet")
    if gt_df is None:
        raise FileNotFoundError(
            f"ground_truth.parquet not found or invalid in {predictions_dir}"
        )
    gt_answer = _build_answer_labels(gt_df)
    gt_token = _build_token_labels(gt_df)

    answer_results: Dict[str, MethodInfo] = {}
    token_results: Dict[str, MethodInfo] = {}

    # Threshold helper
    if thresholds is not None:
        _thr = lambda key: _get_threshold(thresholds, key)
    else:
        _thr = lambda key: 0.5

    # ------------------------------------------------------------------
    # 1. Answer-level baseline methods
    # ------------------------------------------------------------------

    # HaluNet
    df = _load_prediction_df(predictions_dir, "halunet.parquet", ["question_id", "score"])
    if df is not None:
        answer_results["HaluNet"] = ("answer", _eval_answer_level(df, gt_answer, "score", threshold=_thr("halunet")))

    # SemanticUncertainty (defensive: skip if bad qids)
    df = _load_prediction_df(predictions_dir, "semantic_uncertainty.parquet", ["question_id", "score"])
    if df is not None:
        answer_results["SemanticUncertainty"] = ("answer", _eval_answer_level(df, gt_answer, "score", threshold=_thr("semantic_uncertainty")))

    # SemanticEnergy
    df = _load_prediction_df(predictions_dir, "semantic_energy.parquet", ["question_id", "score"])
    if df is not None:
        answer_results["SemanticEnergy"] = ("answer", _eval_answer_level(df, gt_answer, "score", threshold=_thr("semantic_energy")))

    # SelfCheckGPT NLI / Prompt
    for variant in ("nli", "prompt"):
        df = _load_prediction_df(predictions_dir, f"selfcheck_{variant}.parquet", ["question_id", "score"])
        if df is not None:
            name = f"SelfCheckGPT-{variant.upper()}"
            answer_results[name] = ("answer", _eval_answer_level(df, gt_answer, "score", threshold=_thr(f"selfcheck_{variant}")))

    # ------------------------------------------------------------------
    # 2. Token-level → answer-level (mean + max)
    # ------------------------------------------------------------------

    # PredictiveEntropy
    pe_df = _load_prediction_df(predictions_dir, "predictive_entropy.parquet", ["question_id", "token_position", "score"])
    if pe_df is not None:
        for agg in ("mean", "max"):
            agg_df = _aggregate_token_to_answer(pe_df, "score", agg)
            t = _thr(f"predictive_entropy_{agg}")
            answer_results[f"PredictiveEntropy-{agg}"] = (
                "answer",
                _eval_answer_level(agg_df, gt_answer, "score", threshold=t),
            )
        # Token-level
        token_results["PredictiveEntropy"] = (
            "token",
            _eval_token_level(pe_df, gt_token, "score"),
        )

    # LLM-Check (multiple score types)
    llm_df = _load_prediction_df(predictions_dir, "llm_check.parquet", ["question_id", "token_position", "attention_score", "hidden_score", "perplexity_score", "entropy_score"])
    if llm_df is not None:
        for score_type, agg in (
            ("attention_score", "mean"),
            ("attention_score", "max"),
            ("hidden_score", "mean"),
            ("hidden_score", "max"),
            ("entropy_score", "mean"),
            ("entropy_score", "max"),
        ):
            agg_df = _aggregate_token_to_answer(llm_df, score_type, agg)
            name = f"LLM-Check-{score_type.replace('_score', '')}-{agg}"
            key = f"llm_check_{score_type.replace('_score', '')}_{agg}"
            answer_results[name] = ("answer", _eval_answer_level(agg_df, gt_answer, score_type, threshold=_thr(key)))

        # Perplexity is answer-level scalar in llm_check
        answer_results["LLM-Check-perplexity"] = (
            "answer",
            _eval_answer_level(
                llm_df[["question_id", "perplexity_score"]].groupby("question_id", as_index=False).first(),
                gt_answer,
                "perplexity_score",
                threshold=_thr("llm_check_perplexity"),
            ),
        )

        # Token-level for attention, hidden, entropy
        for score_type in ("attention_score", "hidden_score", "entropy_score"):
            token_results[f"LLM-Check-{score_type.replace('_score', '')}"] = (
                "token",
                _eval_token_level(llm_df, gt_token, score_type),
            )

    # Individual MoE Signals (per-token → answer-level mean/max)
    INDIVIDUAL_SIGNALS = [
        "router_entropy",
        "expert_hidden_scores",
        "expert_similarities",
        "expert_usage_entropy",
        "expert_usage_gini",
        "expert_usage_effective_experts",
    ]
    for signal_name in INDIVIDUAL_SIGNALS:
        sig_df = _load_prediction_df(
            predictions_dir, f"signal_{signal_name}.parquet",
            ["question_id", "token_position", "score"],
        )
        if sig_df is None:
            continue
        for agg in ("mean", "max"):
            agg_df = _aggregate_token_to_answer(sig_df, "score", agg)
            name = f"Signal-{signal_name}-{agg}"
            key = f"individual_signal_{signal_name}_{agg}"
            answer_results[name] = (
                "answer", _eval_answer_level(agg_df, gt_answer, "score", threshold=_thr(key))
            )
        token_results[f"Signal-{signal_name}"] = (
            "token", _eval_token_level(sig_df, gt_token, "score")
        )

    # MoE Detector — every candidate family (per-token → answer-level mean/max)
    for det_path in sorted(predictions_dir.glob("detector_*.parquet")):
        det_df = _load_prediction_df(
            predictions_dir, det_path.name,
            ["question_id", "token_position", "score"],
        )
        if det_df is None:
            continue
        family = det_path.stem[len("detector_"):]
        thr = _thr(f"detector_{family}")
        agg_df = _aggregate_token_to_answer(det_df, "score", "mean")
        answer_results[f"Ours-{family} (mean)"] = (
            "answer", _eval_answer_level(agg_df, gt_answer, "score", threshold=thr)
        )
        agg_df = _aggregate_token_to_answer(det_df, "score", "max")
        answer_results[f"Ours-{family} (max)"] = (
            "answer", _eval_answer_level(agg_df, gt_answer, "score", threshold=thr)
        )
        token_results[f"Ours-{family}"] = (
            "token", _eval_token_level(det_df, gt_token, "score")
        )

    # ------------------------------------------------------------------
    # 3. Build comparison tables
    # ------------------------------------------------------------------

    def _make_table(results: Dict[str, MethodInfo]) -> str:
        rows = []
        cols = ["auroc", "auprc", "f1", "accuracy", "tpr_at_1fpr", "tpr_at_5fpr", "tpr_at_10fpr", "ece"]
        display_names = {
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "f1": "F1",
            "accuracy": "Accuracy",
            "tpr_at_1fpr": "TPR@1%FPR",
            "tpr_at_5fpr": "TPR@5%FPR",
            "tpr_at_10fpr": "TPR@10%FPR",
            "ece": "ECE",
        }
        for method, (_, metrics) in results.items():
            row_cells = [method]
            row_cells.extend(
                f"{metrics.get(c, 0.0):.3f}" for c in cols
            )
            rows.append("| " + " | ".join(row_cells) + " |")
        header = "| Method | " + " | ".join(display_names[c] for c in cols) + " |"
        sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
        return "\n".join([header, sep] + rows)

    answer_table = _make_table(answer_results)
    token_table = _make_table(token_results)

    comparison_md = (
        "# Results Comparison\n\n"
        f"## Answer-Level Evaluation ({len(gt_answer)} samples)\n\n"
        f"{answer_table}\n\n"
        f"## Token-Level Evaluation ({len(gt_token)} tokens)\n\n"
        f"{token_table}\n\n"
        "*TPR@X%FPR = True Positive Rate at X% False Positive Rate.  "
        "ECE = Expected Calibration Error (10 bins).*\n"
    )
    (output_dir / "comparison_table.md").write_text(comparison_md)
    print(f"  Saved {output_dir / 'comparison_table.md'}")

    # ------------------------------------------------------------------
    # 3b. CSV exports of the same tables (one row per method)
    # ------------------------------------------------------------------
    csv_cols = [
        "auroc", "auprc", "f1", "accuracy",
        "tpr_at_1fpr", "tpr_at_5fpr", "tpr_at_10fpr", "ece",
        "threshold", "n",
    ]

    def _make_table_df(results: Dict[str, MethodInfo]) -> pd.DataFrame:
        rows = []
        for method, (_, metrics) in results.items():
            row = {"method": method}
            for c in csv_cols:
                row[c] = metrics.get(c, 0.0)
            rows.append(row)
        return pd.DataFrame(rows, columns=["method"] + csv_cols)

    answer_csv = output_dir / "comparison_table_answer.csv"
    token_csv = output_dir / "comparison_table_token.csv"
    _make_table_df(answer_results).to_csv(answer_csv, index=False, float_format="%.6f")
    print(f"  Saved {answer_csv}")
    _make_table_df(token_results).to_csv(token_csv, index=False, float_format="%.6f")
    print(f"  Saved {token_csv}")

    # ------------------------------------------------------------------
    # 4. Save raw results.json
    # ------------------------------------------------------------------
    raw = {
        "answer_level": {
            k: v[1] for k, v in answer_results.items()
        },
        "token_level": {
            k: v[1] for k, v in token_results.items()
        },
    }
    (output_dir / "results.json").write_text(json.dumps(raw, indent=2))
    print(f"  Saved {output_dir / 'results.json'}")

    # ------------------------------------------------------------------
    # 5. ROC curves (answer-level)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted_any = False
    for method, (_, metrics) in answer_results.items():
        # Recompute ROC curve from data
        pred_fn = {
            "HaluNet": "halunet.parquet",
            "SemanticUncertainty": "semantic_uncertainty.parquet",
            "SemanticEnergy": "semantic_energy.parquet",
            "SelfCheckGPT-NLI": "selfcheck_nli.parquet",
            "SelfCheckGPT-PROMPT": "selfcheck_prompt.parquet",
        }.get(method)
        if method.startswith("PredictiveEntropy-"):
            pred_fn = "predictive_entropy.parquet"
        elif method.startswith("LLM-Check-"):
            pred_fn = "llm_check.parquet"
        elif method.startswith("Signal-"):
            pred_fn = f"signal_{method[len('Signal-'):].rsplit('-', 1)[0]}.parquet"
        elif method.startswith("Ours-"):
            pred_fn = _ours_method_to_pred_fn(method)

        if pred_fn is None:
            continue

        pred_df = _load_prediction_df(predictions_dir, pred_fn)
        if pred_df is None:
            continue

        score_col = {
            "LLM-Check-attention-mean": "attention_score",
            "LLM-Check-attention-max": "attention_score",
            "LLM-Check-hidden-mean": "hidden_score",
            "LLM-Check-hidden-max": "hidden_score",
            "LLM-Check-entropy-mean": "entropy_score",
            "LLM-Check-entropy-max": "entropy_score",
            "LLM-Check-perplexity": "perplexity_score",
        }.get(method, "score")

        # Handle aggregation for token-level methods
        if method.startswith(("PredictiveEntropy-", "Ours-", "LLM-Check-attention-", "LLM-Check-hidden-", "LLM-Check-entropy-", "Signal-")):
            # Need to extract aggregation
            if method.endswith("(max)") or "-max" in method:
                agg = "max"
            else:
                agg = "mean"
            pred_df = _aggregate_token_to_answer(pred_df, score_col, agg)
        elif method == "LLM-Check-perplexity":
            pred_df = pred_df[["question_id", "perplexity_score"]].groupby("question_id", as_index=False).first()

        merged = gt_answer.merge(pred_df[["question_id", score_col]], on="question_id", how="inner")
        if len(merged) == 0 or len(np.unique(merged["label"])) < 2:
            continue
        y_proba = np.nan_to_num(merged[score_col].astype(float).values,
                                nan=0.0, posinf=1.0, neginf=0.0)
        fpr, tpr, _ = roc_curve(merged["label"].values, y_proba)
        ax.plot(fpr, tpr, label=f"{method} (AUC={metrics['auroc']:.3f})")
        plotted_any = True

    if plotted_any:
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Answer-Level ROC Curves")
        ax.legend(loc="lower right", fontsize="small")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curves.png", dpi=plot_dpi)
        print(f"  Saved {output_dir / 'roc_curves.png'}")
    else:
        print("  WARNING: No valid data for ROC curves.")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Calibration plots (answer-level)
    # ------------------------------------------------------------------
    n_methods = len(answer_results)
    if n_methods > 0:
        cols = 3
        rows = (n_methods + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        if rows == 1:
            axes = np.array(axes).reshape(1, -1) if n_methods > 1 else np.array([[axes]])
        else:
            axes = axes.reshape(rows, cols)

        for ax in axes.flat:
            ax.set_visible(False)

        for idx, (method, (_, metrics)) in enumerate(answer_results.items()):
            pred_fn = {
                "HaluNet": "halunet.parquet",
                "SemanticUncertainty": "semantic_uncertainty.parquet",
                "SemanticEnergy": "semantic_energy.parquet",
                "SelfCheckGPT-NLI": "selfcheck_nli.parquet",
                "SelfCheckGPT-PROMPT": "selfcheck_prompt.parquet",
            }.get(method)
            if method.startswith("PredictiveEntropy-"):
                pred_fn = "predictive_entropy.parquet"
            elif method.startswith("LLM-Check-"):
                pred_fn = "llm_check.parquet"
            elif method.startswith("Ours-"):
                pred_fn = _ours_method_to_pred_fn(method)

            if pred_fn is None:
                continue

            pred_df = _load_prediction_df(predictions_dir, pred_fn)
            if pred_df is None:
                continue

            score_col = {
                "LLM-Check-attention-mean": "attention_score",
                "LLM-Check-attention-max": "attention_score",
                "LLM-Check-hidden-mean": "hidden_score",
                "LLM-Check-hidden-max": "hidden_score",
                "LLM-Check-entropy-mean": "entropy_score",
                "LLM-Check-entropy-max": "entropy_score",
                "LLM-Check-perplexity": "perplexity_score",
            }.get(method, "score")

            # Handle aggregation for token-level methods
            if method.startswith(("PredictiveEntropy-", "Ours-", "LLM-Check-attention-", "LLM-Check-hidden-", "LLM-Check-entropy-")):
                if method.endswith("(max)") or "-max" in method:
                    pred_df = _aggregate_token_to_answer(pred_df, score_col, "max")
                else:
                    pred_df = _aggregate_token_to_answer(pred_df, score_col, "mean")
            elif method == "LLM-Check-perplexity":
                pred_df = pred_df[["question_id", "perplexity_score"]].groupby("question_id", as_index=False).first()

            if score_col not in pred_df.columns:
                continue

            merged = gt_answer.merge(pred_df[["question_id", score_col]], on="question_id", how="inner")
            if len(merged) == 0:
                continue

            y_true = merged["label"].astype(int).values
            y_proba = np.nan_to_num(merged[score_col].astype(float).values,
                                    nan=0.0, posinf=1.0, neginf=0.0)

            # Normalize scores to [0, 1] for calibration display if needed
            if y_proba.min() < 0 or y_proba.max() > 1:
                # Min-max normalise
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)

            if len(np.unique(y_true)) < 2:
                continue

            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
            ax = axes.flat[idx]
            ax.plot(prob_pred, prob_true, "s-", label="Model")
            ax.plot([0, 1], [0, 1], "k--", label="Perfect")
            ax.set_title(f"{method}\nECE={metrics['ece']:.3f}")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend(loc="lower right", fontsize="xx-small")
            ax.set_visible(True)

        fig.tight_layout()
        fig.savefig(output_dir / "calibration_plots.png", dpi=plot_dpi)
        print(f"  Saved {output_dir / 'calibration_plots.png'}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"  Answer-level methods: {len(answer_results)}")
    print(f"  Token-level methods:  {len(token_results)}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_predictions_dir(
    data_root: Path, model: str, test_years: List[int], test_month: Optional[int]
) -> Path:
    """Replicate the path resolution used by 4.0-model-evaluation.py.

    Returns the directory that actually contains the parquet files.
    """
    model_slug = resolve_model_slug(model)
    _, _, dataset_slug = resolve_dataset_slug(test_years, test_month)
    data_path = data_root / dataset_slug / model_slug

    # --- Direct match first -------------------------------------------------
    pred_dir = data_path / "predictions"
    if _has_predictions(pred_dir):
        return pred_dir

    # --- Combined-dataset fallback -----------------------------------------
    # Scan data_root for any dataset that covers the requested years and
    # actually contains predictions for this model.
    for dataset_dir in data_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        parsed = _parse_dataset_dir_name(dataset_dir.name)
        if parsed is None:
            continue
        candidate_years, candidate_month = parsed
        if test_month is not None:
            exact_match = candidate_month == test_month and set(test_years).issubset(set(candidate_years))
            fallback_match = candidate_month is None and set(test_years).issubset(set(candidate_years))
            if not (exact_match or fallback_match):
                continue
        else:
            if candidate_month is not None:
                continue
            if not set(test_years).issubset(set(candidate_years)):
                continue

        model_dir = dataset_dir / model_slug
        pred_dir = model_dir / "predictions"
        if _has_predictions(pred_dir):
            return pred_dir
        sub = pred_dir / "predictions"
        if _has_predictions(sub):
            return sub

    raise FileNotFoundError(
        f"Could not find predictions for model={model}, years={test_years}, month={test_month}. "
        f"Tried {data_path / 'predictions'} and combined fallbacks."
    )


def _has_predictions(pred_dir: Path) -> bool:
    """Check whether a predictions directory exists and holds ground_truth.parquet."""
    return pred_dir.exists() and (pred_dir / "ground_truth.parquet").exists()


def main():
    parser = argparse.ArgumentParser(
        description="Analyse hallucination detection results (5.0)."
    )

    # Two mutually-exclusive ways to locate predictions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Explicit path to predictions *.parquet files + ground_truth.parquet. "
             "If given, --model/--test-years/--data-root are ignored.",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g. 'allenai/OLMoE-1B-7B-0924-Instruct'). "
             "Used together with --test-years to auto-resolve the predictions directory.",
    )

    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=None,
        help="Test year(s) (e.g. 2026). Required when --model is used.",
    )
    parser.add_argument(
        "--test-month",
        type=int,
        default=None,
        help="Test month (1-12). Optional.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write outputs. Defaults to <predictions_dir>/../analysis.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
    )
    parser.add_argument(
        "--thresholds-file",
        type=Path,
        default=None,
        help="Optional path to thresholds.json (produced by 3.1-fit-baselines.py). "
             "If provided, F1/Accuracy/TPR@X%FPR/ECE are computed at the tuned threshold "
             "for each baseline instead of 0.5.",
    )
    args = parser.parse_args()

    # Resolve predictions directory
    if args.predictions_dir is not None:
        predictions_dir = args.predictions_dir
        print(f"Using explicit predictions dir: {predictions_dir}")
    else:
        if args.test_years is None or len(args.test_years) == 0:
            parser.error("--test-years is required when using --model")
        predictions_dir = _resolve_predictions_dir(
            args.data_root, args.model, args.test_years, args.test_month
        )
        print(f"Resolved predictions dir: {predictions_dir}")

    # Resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        # Place it next to the outer "predictions" folder so the layout is
        #   <model_slug>/predictions/   (parquet files)
        #   <model_slug>/analysis/      (results)
        if predictions_dir.parent.name == "predictions":
            output_dir = predictions_dir.parent.parent / "analysis"
        else:
            output_dir = predictions_dir.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Load thresholds if provided
    thresholds: Optional[Dict[str, Dict]] = None
    if args.thresholds_file is not None and args.thresholds_file.exists():
        with open(args.thresholds_file, "r") as f:
            thresholds = json.load(f)
        print(f"Loaded thresholds from {args.thresholds_file}")

        models_dir = args.thresholds_file.parent

        # HaluNet threshold is stored separately in halunet_train_summary.json
        halunet_summary_path = models_dir / "halunet_train_summary.json"
        if halunet_summary_path.exists():
            with open(halunet_summary_path, "r") as f:
                halunet_summary = json.load(f)
            halunet_thr = halunet_summary.get("optimal_threshold")
            if halunet_thr is not None:
                thresholds = dict(thresholds)
                thresholds["halunet"] = {"threshold": float(halunet_thr)}
                print(f"  Also loaded HaluNet threshold: {halunet_thr}")

        # Detector thresholds are stored inside the detector pickles.
        # Load the overall-best detector.pkl plus every per-family
        # detector_<Family>.pkl written by 3.0-detection-model-training.py.
        sys.path.append(str(Path(__file__).parent.parent))
        from moeuncert.experiments import replace_inf_with_nan
        # Backward-compat alias so pickles trained before _replace_inf_with_nan
        # moved to moeuncert.experiments.utils can still resolve the function.
        sys.modules["__main__"]._replace_inf_with_nan = replace_inf_with_nan

        detector_paths = [models_dir / "detector.pkl"]
        detector_paths += sorted(models_dir.glob("detector_*.pkl"))
        for detector_path in detector_paths:
            if not detector_path.exists():
                continue
            try:
                with open(detector_path, "rb") as f:
                    detector = pickle.load(f)
            except Exception as e:
                print(f"  WARNING: Could not load {detector_path.name}: {e}")
                continue
            det_thr = detector.get("optimal_threshold")
            if det_thr is None:
                continue
            if detector_path.stem.startswith("detector_"):
                key = f"detector_{detector_path.stem[len('detector_'):]}"
            else:
                key = "detector"
            thresholds = dict(thresholds)
            thresholds[key] = {"threshold": float(det_thr)}
            print(f"  Also loaded {key} threshold: {det_thr}")
    elif args.thresholds_file is not None:
        print(f"WARNING: thresholds file not found: {args.thresholds_file}")

    analyse(predictions_dir, output_dir, args.plot_dpi, thresholds=thresholds)


if __name__ == "__main__":
    main()
