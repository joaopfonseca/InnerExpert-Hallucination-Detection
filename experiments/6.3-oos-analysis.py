"""8.0 — Cross-dataset OOS analysis: per-method metrics across all datasets.

Reads ``predictions/`` parquets + ``ground_truth.parquet`` from every
OOS dataset directory, computes per-dataset × per-method metrics
(AUROC / AUPRC / F1 / Accuracy / TPR@1/5/10%FPR / ECE), and produces
a cross-dataset comparison table plus aggregated summary.

Reuses the metric function from ``moeuncert.evaluation.metrics`` (also
used by 5.0 for the in-distribution analysis).

Outputs (to ``--output-dir``, default ``data/oos-comparison/{model_slug}/``):
  - comparison_table.md          — Markdown table, grouped by dataset
  - comparison_table_answer.csv  — One row per (dataset, method)
  - comparison_table_token.csv   — One row per (dataset, method)
  - cross_dataset_summary.md      — One row per method, averaged across datasets
  - roc_curves.png                — One subplot per dataset
  - calibration_plots.png         — Reliability diagrams
  - results.json                   — Raw numbers

Usage:
    python 8.0-oos-analysis.py --datasets squad truthfulqa nq_open freshqa
    python 8.0-oos-analysis.py --model allenai/OLMoE-1B-7B-0924-Instruct \
        --thresholds-file models/allenai__OLMoE-1B-7B-0924-Instruct/thresholds.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.datasets_adapters import OOS_ADAPTERS, get_adapter, list_oos_datasets
from moeuncert.experiments import resolve_model_slug
from moeuncert.evaluation.metrics import compute_metrics_for_predictions


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_prediction_df(predictions_dir: Path, filename: str, required_cols=None):
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
    return df


def _build_answer_labels(gt_df: pd.DataFrame) -> pd.DataFrame:
    return (
        gt_df[["question_id", "answer_label"]]
        .groupby("question_id", as_index=False)
        .first()
        .rename(columns={"answer_label": "label"})
    )


def _build_token_labels(gt_df: pd.DataFrame) -> pd.DataFrame:
    return gt_df[["question_id", "token_position", "token_label"]].rename(
        columns={"token_label": "label"}
    )


def _eval_answer_level(pred_df, gt_answer, score_col, label_col="label", threshold=0.5):
    merged = gt_answer[["question_id", label_col]].merge(
        pred_df[["question_id", score_col]],
        on="question_id", how="inner",
    )
    if len(merged) == 0:
        return compute_metrics_for_predictions(np.array([]), np.array([]))
    y_true = merged[label_col].astype(int).values
    y_proba = merged[score_col].astype(float).values
    return compute_metrics_for_predictions(y_true, y_proba, threshold)


def _eval_token_level(pred_df, gt_token, score_col, threshold=0.5):
    merged = gt_token[["question_id", "token_position", "label"]].merge(
        pred_df[["question_id", "token_position", score_col]],
        on=["question_id", "token_position"], how="inner",
    )
    if len(merged) == 0:
        return compute_metrics_for_predictions(np.array([]), np.array([]))
    y_true = merged["label"].astype(int).values
    y_proba = merged[score_col].astype(float).values
    return compute_metrics_for_predictions(y_true, y_proba, threshold)


def _aggregate_token_to_answer(token_df, score_col, agg="mean"):
    grouped = token_df.groupby("question_id", as_index=False)
    if agg == "mean":
        return grouped[score_col].mean()
    elif agg == "max":
        return grouped[score_col].max()
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


# ---------------------------------------------------------------------------
# Per-dataset analysis (mirrors 5.0's analyse() but for a single dataset)
# ---------------------------------------------------------------------------


def analyse_dataset(
    predictions_dir: Path,
    thresholds: Optional[Dict[str, Dict]] = None,
) -> Tuple[Dict[str, Tuple[str, Dict]], Dict[str, Tuple[str, Dict]]]:
    """Analyse one dataset's predictions directory.

    Returns (answer_results, token_results) where each is a dict
    mapping method_name → (level, metrics_dict).
    """
    gt_df = _load_prediction_df(predictions_dir, "ground_truth.parquet")
    if gt_df is None:
        raise FileNotFoundError(
            f"ground_truth.parquet not found or invalid in {predictions_dir}"
        )
    gt_answer = _build_answer_labels(gt_df)
    gt_token = _build_token_labels(gt_df)

    answer_results: Dict[str, Tuple[str, Dict]] = {}
    token_results: Dict[str, Tuple[str, Dict]] = {}

    _thr = (lambda key: _get_threshold(thresholds, key)) if thresholds else (lambda key: 0.5)

    # --- HaluNet ---
    df = _load_prediction_df(predictions_dir, "halunet.parquet", ["question_id", "score"])
    if df is not None:
        answer_results["HaluNet"] = ("answer", _eval_answer_level(df, gt_answer, "score", threshold=_thr("halunet")))

    # --- SemanticUncertainty ---
    df = _load_prediction_df(predictions_dir, "semantic_uncertainty.parquet", ["question_id", "score"])
    if df is not None:
        answer_results["SemanticUncertainty"] = ("answer", _eval_answer_level(df, gt_answer, "score", threshold=_thr("semantic_uncertainty")))

    # --- SemanticEnergy ---
    df = _load_prediction_df(predictions_dir, "semantic_energy.parquet", ["question_id", "score"])
    if df is not None:
        answer_results["SemanticEnergy"] = ("answer", _eval_answer_level(df, gt_answer, "score", threshold=_thr("semantic_energy")))

    # --- SelfCheckGPT ---
    for variant in ("nli", "prompt"):
        df = _load_prediction_df(predictions_dir, f"selfcheck_{variant}.parquet", ["question_id", "score"])
        if df is not None:
            answer_results[f"SelfCheckGPT-{variant.upper()}"] = ("answer", _eval_answer_level(df, gt_answer, "score", threshold=_thr(f"selfcheck_{variant}")))

    # --- PredictiveEntropy ---
    pe_df = _load_prediction_df(predictions_dir, "predictive_entropy.parquet", ["question_id", "token_position", "score"])
    if pe_df is not None:
        for agg in ("mean", "max"):
            agg_df = _aggregate_token_to_answer(pe_df, "score", agg)
            t = _thr(f"predictive_entropy_{agg}")
            answer_results[f"PredictiveEntropy-{agg}"] = ("answer", _eval_answer_level(agg_df, gt_answer, "score", threshold=t))
        token_results["PredictiveEntropy"] = ("token", _eval_token_level(pe_df, gt_token, "score"))

    # --- LLM-Check ---
    llm_df = _load_prediction_df(predictions_dir, "llm_check.parquet",
                                 ["question_id", "token_position", "attention_score", "hidden_score", "perplexity_score", "entropy_score"])
    if llm_df is not None:
        for score_type, agg in (
            ("attention_score", "mean"), ("attention_score", "max"),
            ("hidden_score", "mean"), ("hidden_score", "max"),
            ("entropy_score", "mean"), ("entropy_score", "max"),
        ):
            agg_df = _aggregate_token_to_answer(llm_df, score_type, agg)
            name = f"LLM-Check-{score_type.replace('_score', '')}-{agg}"
            key = f"llm_check_{score_type.replace('_score', '')}_{agg}"
            answer_results[name] = ("answer", _eval_answer_level(agg_df, gt_answer, score_type, threshold=_thr(key)))

        answer_results["LLM-Check-perplexity"] = (
            "answer",
            _eval_answer_level(
                llm_df[["question_id", "perplexity_score"]].groupby("question_id", as_index=False).first(),
                gt_answer, "perplexity_score", threshold=_thr("llm_check_perplexity"),
            ),
        )

        for score_type in ("attention_score", "hidden_score", "entropy_score"):
            token_results[f"LLM-Check-{score_type.replace('_score', '')}"] = (
                "token", _eval_token_level(llm_df, gt_token, score_type),
            )

    # --- MoE Detector ---
    for det_path in sorted(predictions_dir.glob("detector_*.parquet")):
        det_df = _load_prediction_df(predictions_dir, det_path.name,
                                     ["question_id", "token_position", "score"])
        if det_df is None:
            continue
        family = det_path.stem[len("detector_"):]
        thr = _thr(f"detector_{family}")
        for agg in ("mean", "max"):
            agg_df = _aggregate_token_to_answer(det_df, "score", agg)
            answer_results[f"Ours-{family} ({agg})"] = (
                "answer", _eval_answer_level(agg_df, gt_answer, "score", threshold=thr),
            )
        token_results[f"Ours-{family}"] = ("token", _eval_token_level(det_df, gt_token, "score"))

    return answer_results, token_results


def _get_threshold(thresholds: Optional[Dict[str, Dict]], key: str) -> float:
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


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------


METRIC_COLS = ["auroc", "auprc", "f1", "accuracy", "tpr_at_1fpr", "tpr_at_5fpr", "tpr_at_10fpr", "ece"]
DISPLAY_NAMES = {
    "auroc": "AUROC", "auprc": "AUPRC", "f1": "F1", "accuracy": "Accuracy",
    "tpr_at_1fpr": "TPR@1%FPR", "tpr_at_5fpr": "TPR@5%FPR", "tpr_at_10fpr": "TPR@10%FPR",
    "ece": "ECE",
}


def _make_table_df(results: Dict[str, Tuple[str, Dict]], dataset_name: str) -> pd.DataFrame:
    rows = []
    for method, (_, metrics) in results.items():
        row = {"dataset": dataset_name, "method": method, "level": results[method][0]}
        for c in METRIC_COLS + ["threshold", "n"]:
            row[c] = metrics.get(c, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_markdown_table(results: Dict[str, Tuple[str, Dict]], title: str) -> str:
    if not results:
        return f"{title}\n\n_No data._\n"
    lines = [title, ""]
    header = "| Method | " + " | ".join(DISPLAY_NAMES[c] for c in METRIC_COLS) + " |"
    sep = "|" + "|".join(["---"] * (len(METRIC_COLS) + 1)) + "|"
    lines.extend([header, sep])
    for method, (_, metrics) in results.items():
        row_cells = [method]
        row_cells.extend(f"{metrics.get(c, 0.0):.3f}" for c in METRIC_COLS)
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset OOS analysis (8.0)"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=list_oos_datasets(),
        help="OOS dataset names to analyse (default: all)",
    )
    parser.add_argument(
        "--model", type=str, default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="Subject model name",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("models"),
        help="Directory with trained models (for thresholds.json)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for comparison tables + plots (default: data/oos-comparison/{model_slug})",
    )
    parser.add_argument(
        "--thresholds-file", type=Path, default=None,
        help="Optional path to thresholds.json (from 3.1-fit-baselines.py)",
    )
    parser.add_argument(
        "--plot-dpi", type=int, default=300,
        help="DPI for saved figures",
    )

    args = parser.parse_args()

    print(f"{'=' * 70}")
    print("8.0 — OOS CROSS-DATASET ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Datasets: {args.datasets}")
    print(f"Model: {args.model}")

    model_slug = resolve_model_slug(args.model)

    # Resolve output directory (default: model-specific subdir)
    if args.output_dir is None:
        args.output_dir = Path("data") / "oos-comparison" / model_slug
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load thresholds (optional) --------------------------------------
    thresholds: Optional[Dict[str, Dict]] = None
    if args.thresholds_file is not None and args.thresholds_file.exists():
        with open(args.thresholds_file, "r") as f:
            thresholds = json.load(f)
        print(f"Loaded thresholds from {args.thresholds_file}")

        # HaluNet threshold
        halunet_summary_path = args.thresholds_file.parent / "halunet_train_summary.json"
        if halunet_summary_path.exists():
            with open(halunet_summary_path, "r") as f:
                halunet_summary = json.load(f)
            halunet_thr = halunet_summary.get("optimal_threshold")
            if halunet_thr is not None:
                thresholds = dict(thresholds)
                thresholds["halunet"] = {"threshold": float(halunet_thr)}

        # Detector thresholds from pickles
        sys.path.append(str(Path(__file__).parent.parent))
        from moeuncert.experiments import replace_inf_with_nan
        sys.modules["__main__"]._replace_inf_with_nan = replace_inf_with_nan

        for detector_path in [args.thresholds_file.parent / "detector.pkl"] + \
                sorted(args.thresholds_file.parent.glob("detector_*.pkl")):
            if not detector_path.exists():
                continue
            try:
                with open(detector_path, "rb") as f:
                    detector = pickle.load(f)
            except Exception:
                continue
            det_thr = detector.get("optimal_threshold")
            if det_thr is None:
                continue
            if detector_path.stem.startswith("detector_"):
                key = f"detector_{detector_path.stem[len('detector_'):]}"
            else:
                key = "detector"
            thresholds[key] = {"threshold": float(det_thr)}

    # --- Analyse each dataset ---------------------------------------------
    all_answer_dfs = []
    all_token_dfs = []
    per_dataset_results: Dict[str, Tuple[Dict, Dict]] = {}

    for ds_name in args.datasets:
        adapter = get_adapter(ds_name)
        ds_dir = args.data_root / adapter.slug / model_slug
        predictions_dir = ds_dir / "predictions"

        if not predictions_dir.exists() or not (predictions_dir / "ground_truth.parquet").exists():
            print(f"\n  SKIP {ds_name}: no predictions at {predictions_dir}")
            continue

        print(f"\nAnalysing {ds_name}...")
        answer_results, token_results = analyse_dataset(predictions_dir, thresholds=thresholds)
        per_dataset_results[ds_name] = (answer_results, token_results)

        all_answer_dfs.append(_make_table_df(answer_results, ds_name))
        all_token_dfs.append(_make_table_df(token_results, ds_name))

        print(f"  Answer-level methods: {len(answer_results)}")
        print(f"  Token-level methods: {len(token_results)}")

    if not per_dataset_results:
        print("\nNo datasets with predictions found. Nothing to analyse.")
        return

    # --- Build comparison tables ------------------------------------------
    answer_csv = pd.concat(all_answer_dfs, ignore_index=True)
    token_csv = pd.concat(all_token_dfs, ignore_index=True)

    answer_csv_path = args.output_dir / "comparison_table_answer.csv"
    token_csv_path = args.output_dir / "comparison_table_token.csv"
    answer_csv.to_csv(answer_csv_path, index=False, float_format="%.6f")
    token_csv.to_csv(token_csv_path, index=False, float_format="%.6f")
    print(f"\nSaved {answer_csv_path}")
    print(f"Saved {token_csv_path}")

    # Markdown table (grouped by dataset)
    md_parts = ["# OOS Cross-Dataset Comparison\n"]
    for ds_name, (answer_results, token_results) in per_dataset_results.items():
        adapter = get_adapter(ds_name)
        n_gt = len(_load_prediction_df(
            args.data_root / adapter.slug / model_slug / "predictions",
            "ground_truth.parquet",
        ))
        md_parts.append(f"## {ds_name} ({n_gt} ground-truth tokens)\n")
        md_parts.append(_make_markdown_table(answer_results, "### Answer-Level"))
        md_parts.append(_make_markdown_table(token_results, "### Token-Level"))

    md_parts.append(
        "\n*TPR@X%FPR = True Positive Rate at X% False Positive Rate. "
        "ECE = Expected Calibration Error (10 bins).*\n"
    )
    (args.output_dir / "comparison_table.md").write_text("\n".join(md_parts))
    print(f"Saved {args.output_dir / 'comparison_table.md'}")

    # --- Cross-dataset summary (averaged across datasets) -----------------
    summary_rows = []
    # Collect all method names across datasets
    all_answer_methods = set()
    all_token_methods = set()
    for answer_results, token_results in per_dataset_results.values():
        all_answer_methods.update(answer_results.keys())
        all_token_methods.update(token_results.keys())

    for method in sorted(all_answer_methods):
        row = {"method": method, "level": "answer"}
        for c in METRIC_COLS:
            vals = [
                per_dataset_results[ds][0][method][1].get(c, np.nan)
                for ds in per_dataset_results
                if method in per_dataset_results[ds][0]
            ]
            row[c] = float(np.nanmean(vals)) if vals else np.nan
            row[f"{c}_std"] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    for method in sorted(all_token_methods):
        row = {"method": method, "level": "token"}
        for c in METRIC_COLS:
            vals = [
                per_dataset_results[ds][1][method][1].get(c, np.nan)
                for ds in per_dataset_results
                if method in per_dataset_results[ds][1]
            ]
            row[c] = float(np.nanmean(vals)) if vals else np.nan
            row[f"{c}_std"] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = args.output_dir / "cross_dataset_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, float_format="%.6f")
    print(f"Saved {summary_csv_path}")

    # Markdown summary
    summary_md = ["# Cross-Dataset Summary (averaged across datasets)\n"]
    for level in ("answer", "token"):
        level_df = summary_df[summary_df["level"] == level]
        if level_df.empty:
            continue
        summary_md.append(f"## {level.capitalize()}-Level\n")
        header = "| Method | " + " | ".join(DISPLAY_NAMES[c] for c in METRIC_COLS) + " |"
        sep = "|" + "|".join(["---"] * (len(METRIC_COLS) + 1)) + "|"
        summary_md.extend([header, sep])
        for _, row in level_df.iterrows():
            cells = [row["method"]]
            cells.extend(f"{row[c]:.3f} ± {row[f'{c}_std']:.3f}" for c in METRIC_COLS)
            summary_md.append("| " + " | ".join(cells) + " |")
        summary_md.append("")

    (args.output_dir / "cross_dataset_summary.md").write_text("\n".join(summary_md))
    print(f"Saved {args.output_dir / 'cross_dataset_summary.md'}")

    # --- results.json -----------------------------------------------------
    raw = {}
    for ds_name, (answer_results, token_results) in per_dataset_results.items():
        raw[ds_name] = {
            "answer_level": {k: v[1] for k, v in answer_results.items()},
            "token_level": {k: v[1] for k, v in token_results.items()},
        }
    (args.output_dir / "results.json").write_text(json.dumps(raw, indent=2))
    print(f"Saved {args.output_dir / 'results.json'}")

    # --- ROC curves (one subplot per dataset) ----------------------------
    n_datasets = len(per_dataset_results)
    if n_datasets > 0:
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_datasets == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(rows, cols)

        for ax in axes.flat:
            ax.set_visible(False)

        for idx, (ds_name, (answer_results, _)) in enumerate(per_dataset_results.items()):
            ax = axes.flat[idx]
            adapter = get_adapter(ds_name)
            predictions_dir = args.data_root / adapter.slug / model_slug / "predictions"
            gt_df = _load_prediction_df(predictions_dir, "ground_truth.parquet")
            gt_answer = _build_answer_labels(gt_df)

            plotted = False
            for method, (_, metrics) in answer_results.items():
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
                    family = method[len("Ours-"):]
                    for suffix in (" (mean)", " (max)"):
                        if family.endswith(suffix):
                            family = family[:-len(suffix)]
                            break
                    pred_fn = f"detector_{family}.parquet"

                if pred_fn is None:
                    continue

                pred_df = _load_prediction_df(predictions_dir, pred_fn)
                if pred_df is None:
                    continue

                score_col = "score"
                if method.startswith("LLM-Check-"):
                    score_col = {
                        "LLM-Check-attention": "attention_score",
                        "LLM-Check-hidden": "hidden_score",
                        "LLM-Check-entropy": "entropy_score",
                        "LLM-Check-perplexity": "perplexity_score",
                    }.get(method.rsplit("-", 1)[0], "score")

                if method.startswith(("PredictiveEntropy-", "Ours-", "LLM-Check-attention-", "LLM-Check-hidden-", "LLM-Check-entropy-")):
                    agg = "max" if (method.endswith("(max)") or "-max" in method) else "mean"
                    pred_df = _aggregate_token_to_answer(pred_df, score_col, agg)
                elif method == "LLM-Check-perplexity":
                    pred_df = pred_df[["question_id", "perplexity_score"]].groupby("question_id", as_index=False).first()
                    score_col = "perplexity_score"

                merged = gt_answer.merge(pred_df[["question_id", score_col]], on="question_id", how="inner")
                if len(merged) == 0 or len(np.unique(merged["label"])) < 2:
                    continue
                scores = np.nan_to_num(
                    merged[score_col].values.astype(np.float64),
                    nan=0.0, posinf=1e10, neginf=0.0,
                )
                fpr, tpr, _ = roc_curve(merged["label"].values, scores)
                ax.plot(fpr, tpr, label=f"{method} (AUC={metrics['auroc']:.3f})")
                plotted = True

            if plotted:
                ax.plot([0, 1], [0, 1], "k--", label="Random")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title(ds_name)
                ax.legend(loc="lower right", fontsize="xx-small")
                ax.set_visible(True)

        fig.tight_layout()
        fig.savefig(args.output_dir / "roc_curves.png", dpi=args.plot_dpi)
        print(f"Saved {args.output_dir / 'roc_curves.png'}")
        plt.close(fig)

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 70}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()