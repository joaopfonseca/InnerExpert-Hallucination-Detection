"""
8.0 — Compile all experiment results into paper-ready plots and tables.

Reads prediction parquets, labeled data, and analysis outputs from both
RealtimeQA-2026 (in-distribution) and OOS datasets (SQuAD, TruthfulQA,
NQ-Open, FreshQA), for both host models (OLMoE-1B-7B, Gemma-4-26B).

Produces LaTeX tables and matplotlib figures for the paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlresearch.latex import export_table, format_table, make_bold
from mlresearch.utils import set_matplotlib_style
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

DATASETS = {
    "squad": "SQuAD", 
    "truthfulqa": "TruthfulQA", 
    "nq_open": "NQ-Open", 
    "freshqa": "FreshQA",
    "realtimeqa-2026": "RTQA"
}
RTQA_DIR = DATA_DIR / "realtimeqa-2026"
HUMAN_VALIDATION_PATH = PROJECT_ROOT / "analysis" / "ANNOTATED_human_validation_sample.xlsx"

# Host models display names
MODELS = {
    "allenai__OLMoE-1B-7B-0924-Instruct": "OLMoE-1B-7B",
    "google__gemma-4-26B-A4B-it": "Gemma-4-26B",
}
MODEL_SLUGS = list(MODELS.keys())


# Method display names
METHOD_NAMES = {

    # Drop max since mean does better in these cases
    "PredictiveEntropy-max": "DROP",
    "LLM-Check-attention-max": "DROP",
    "LLM-Check-hidden-max": "DROP",

    # Token level
    "Ours-XGBoost": "InnerExpert (XGBoost)",
    "Ours-LogisticRegression": "InnerExpert (LR)",
    "Ours-MLP": "InnerExpert (MLP)",
    "Ours-RandomForest": "InnerExpert (RF)",
    "Ours-Transformer": "InnerExpert (Transformer)",
    "PredictiveEntropy": "Logit Entropy",
    "LLM-Check-attention": "LLM-Check (att.)",
    "LLM-Check-hidden": "LLM-Check (hid.)",
    # "LLM-Check-entropy": "DROP",  # "Entropy (LLM-Check)",

    # Answer level
    # "Ours-XGBoost (mean)": "InnerExpert (XGBoost) (mean)",
    # "Ours-LogisticRegression (mean)": "InnerExpert (LR) (mean)",
    # "Ours-MLP (mean)": "InnerExpert (MLP) (mean)",
    # "Ours-RandomForest (mean)": "InnerExpert (RF) (mean)",
    # "Ours-Transformer (mean)": "InnerExpert (Transformer) (mean)",
    # "Ours-XGBoost (max)": "InnerExpert (XGBoost) (max)",
    # "Ours-LogisticRegression (max)": "InnerExpert (LR) (max)",
    # "Ours-MLP (max)": "InnerExpert (MLP) (max)",
    # "Ours-RandomForest (max)": "InnerExpert (RF) (max)",
    # "Ours-Transformer (max)": "InnerExpert (Transformer) (max)",
    "HaluNet": "HaluNet",
    "SemanticUncertainty": "Semantic Uncertainty",
    "SemanticEnergy": "Semantic Energy",
    "SelfCheckGPT-NLI": "SelfCheckGPT (NLI)",
    "SelfCheckGPT-PROMPT": "SelfCheckGPT (Prompt)",
    # "PredictiveEntropy-mean": "Logit Entropy (mean)",
    # "LLM-Check-attention-mean": "LLM-Check (Attention Score) (mean)",


    # "LLM-Check-hidden-mean": "LLM-Check (Hidden Score) (mean)",
    # "LLM-Check-entropy-mean": "DROP",  # "Entropy (LLM-Check) (mean)",
    # "LLM-Check-entropy-max": "DROP",  # "Entropy (LLM-Check) (max)",
    "LLM-Check-perplexity": "Perplexity",
}

# Metrics to report
METRICS = {
    "auroc": "AUROC", 
    "auprc": "AUPRC",
    "f1": "F1",
    "accuracy": "OA",
    "tpr_at_1fpr": "TPR@1%FPR",
    "tpr_at_5fpr": "TPR@5%FPR",
    "tpr_at_10fpr": "TPR@10%FPR",
    # "ece": "ECE",
}
METRICS_INV = {v: k for k, v in METRICS.items()}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
set_matplotlib_style(font_size=8, use_latex=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_oos_comparison(model_slug: str, level: str = "answer") -> pd.DataFrame:
    """Load the cross-dataset comparison CSV for a model (answer or token level)."""
    path = DATA_DIR / "oos-comparison" / model_slug / f"comparison_table_{level}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Comparison table not found: {path}")
    df = pd.read_csv(path)
    return df


def load_rtqa_analysis(model_slug: str, level: str = "answer") -> pd.DataFrame:
    """Load the RealtimeQA-2026 analysis results for a model (answer or token level)."""
    path = RTQA_DIR / model_slug / "analysis" / f"comparison_table_{level}.csv"
    if not path.exists():
        raise FileNotFoundError(f"RTQA analysis not found: {path}")
    df = pd.read_csv(path)
    return df


# def load_predictions(dataset: str, model_slug: str, method: str) -> pd.DataFrame:
#     """Load prediction parquet for a specific dataset/model/method."""
#     if dataset == "realtimeqa-2026":
#         base = RTQA_DIR / model_slug / "predictions"
#     else:
#         base = DATA_DIR / f"oos-{dataset}" / model_slug / "predictions"
#     path = base / f"{method}.parquet"
#     if not path.exists():
#         raise FileNotFoundError(f"Predictions not found: {path}")
#     return pd.read_parquet(path)
# 
# 
# def load_labeled(dataset: str, model_slug: str) -> pd.DataFrame:
#     """Load labeled data for a dataset/model."""
#     if dataset == "realtimeqa-2026":
#         base = RTQA_DIR / model_slug
#     else:
#         base = DATA_DIR / f"oos-{dataset}" / model_slug
#     path = base / "results_labeled_zai-org__GLM-5.1.parquet"
#     if not path.exists():
#         raise FileNotFoundError(f"Labeled data not found: {path}")
#     return pd.read_parquet(path)


def _method_name(method: str, drop_unmapped=True) -> str:
    """Map method slug to display name, or return original if not found."""
    for key, value in METHOD_NAMES.items():
        if method == key: 
            return value
        elif method.startswith(key):
            return f"{value} {method[len(key):]}"

    if drop_unmapped:
        return "DROP"
    else:
        return method


def compile_results(level: str = "answer", drop_unmapped=True) -> pd.DataFrame:
    """Load and combine OOS + RTQA comparison tables for all host models.

    Shared by answer-level and token-level compilation. The comparison tables
    share an identical schema (``dataset``, ``method``, metric columns, plus a
    few extras that are dropped here); only the ``level`` filename suffix
    (``_answer``/``_token``) differs.
    """
    columns = ["dataset", "method"] + list(METRICS.keys())
    combined_dfs = []
    for model_slug, model_name in MODELS.items():
        print(f"\nProcessing model: {model_name} ({model_slug}) [{level}]")
        df_oos = load_oos_comparison(model_slug, level=level)
        df_rtqa = load_rtqa_analysis(model_slug, level=level)

        col_dropped = set(
            [
                col
                for _df in [df_oos, df_rtqa]
                for col in _df.columns[~_df.columns.isin(columns)]
            ]
        )
        print(f"{model_name}: Dropping columns {col_dropped} from results tables")
        df_rtqa["dataset"] = "realtimeqa-2026"
        df_combined = pd.concat([df_oos, df_rtqa], ignore_index=True)[columns]
        df_combined["model"] = model_name
        combined_dfs.append(df_combined)
    df_results = pd.concat(combined_dfs, ignore_index=True)
    df_check = df_results.copy()
    df_results["dataset"] = df_results["dataset"].map(DATASETS)
    df_results["method"] = df_results["method"].apply(
        lambda x: (
            _method_name(x, drop_unmapped=drop_unmapped)
            .replace("-mean", "(mean)")
            .replace(" (max)", "")
            .strip()
        )
    )
    print(
        f"Dropping methods {list(df_check['method'][df_results['method'] == 'DROP'].unique())}"
        "from results table"
    )
    df_results = df_results[df_results["method"] != "DROP"]
    df_results = df_results.sort_values(by=["dataset", "model", "method"])
    df_results.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "method": "Method",
            **METRICS
        },
        inplace=True
    )
    return df_results


def results_table(df_results: pd.DataFrame, metric="F1") -> pd.DataFrame:
    # Generate pivot table for F1 scores (mirrors answer_level_results_table)
    df_main = df_results[
        ~df_results["Method"].map(lambda x: x.endswith("mean)") and x.startswith("InnerExpert"))
    ].copy()
    df_main = df_main[["Dataset", "Model", "Method", metric]].pivot_table(
        index=["Model", "Method"],
        columns="Dataset",
        values=metric
    )
    # compute average ranking across datasets and host models
    ranks = (
        df_main
        .reset_index()
        .drop(columns="Method")
        .groupby(["Model"])
        .rank(ascending=False)
        .mean(axis=1)
    )
    avgs = df_main.mean(axis=1).round(3)
    df_main = df_main.round(3)
    df_main["Avg. Rank"] = ranks.values
    df_main[f"Avg. {metric}"] = avgs.values

    return df_main

def format_results_table_for_export(df: pd.DataFrame) -> pd.DataFrame:
    rank = df.groupby(df.index.get_level_values(0)).apply(lambda _df: make_bold(_df["Avg. Rank"].to_frame(), axis=0, decimals=1, maximum=False))
    df_bold = df.groupby(df.index.get_level_values(0)).apply(lambda _df: make_bold(_df, axis=0, decimals=3))
    df_bold["Avg. Rank"] = rank.values.squeeze()
    return df_bold


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------
LABEL_MODEL_SLUG = "zai-org__GLM-5.1"


def _labeled_parquet_path(dataset_key: str, model_slug: str) -> Path:
    """Resolve the labeled parquet path for a dataset key and model slug."""
    if dataset_key == "realtimeqa-2026":
        base = DATA_DIR / "realtimeqa-2026"
    else:
        base = DATA_DIR / f"oos-{dataset_key}"
    return base / model_slug / f"results_labeled_{LABEL_MODEL_SLUG}.parquet"


def generate_dataset_statistics_table() -> pd.DataFrame:
    """Produce a LaTeX table with basic dataset statistics for all
    evaluation datasets and both host models.

    Columns: Dataset, Model, N, Base, Evidence, Halluc., Grounded,
    Halluc.\\%, Mean Length.
    """
    rows = []
    for ds_key, ds_name in DATASETS.items():
        for model_slug, model_name in MODELS.items():
            path = _labeled_parquet_path(ds_key, model_slug)
            if not path.exists():
                print(f"  WARNING: {path} not found, skipping")
                continue
            df = pd.read_parquet(path)
            n = len(df)
            n_base = int((df["evidence_present"] == 0).sum())
            n_evidence = int((df["evidence_present"] == 1).sum())
            labels = df["label_llm_answer"].dropna().astype(int)
            n_hall = int(labels.sum())
            n_grounded = len(labels) - n_hall
            hall_pct = 100.0 * n_hall / len(labels) if len(labels) > 0 else 0.0
            mean_len = float(df["generated_answer"].str.split().apply(len).mean())
            rows.append({
                "Dataset": ds_name,
                "Model": model_name,
                "N": n,
                "Base": n_base,
                "Evidence": n_evidence,
                "Halluc.": n_hall,
                "Grounded": n_grounded,
                r"Halluc.\,\%": round(hall_pct, 1),
                "Mean Length": round(mean_len, 1),
            })

    df_stats = pd.DataFrame(rows)
    df_stats.to_latex(
        TABLES_DIR / "dataset_statistics.tex",
        index=False,
        escape=False,
        float_format="%.1f",
        caption=(
            "Dataset statistics for all evaluation datasets and host models. "
            "Labels are from the LLM-as-a-judge approach (GLM-5.1). "
            "``Mean Length'' is the average number of words in the generated "
            "answer."
        ),
        label="tab:dataset_statistics",
        column_format="ll" + "c" * (len(df_stats.columns) - 2),
    )
    print(f"  -> {TABLES_DIR / 'dataset_statistics.tex'}")
    return df_stats


# ---------------------------------------------------------------------------
# Signal contribution analysis
# ---------------------------------------------------------------------------

# Answer-level: raw method -> (display_name, aggregation)
SIGNAL_METHOD_NAMES_ANSWER = {
    # MoE individual signals
    "Signal-router_entropy-mean": ("Router Entropy", "mean"),
    "Signal-router_entropy-max": ("Router Entropy", "max"),
    "Signal-expert_hidden_scores-mean": ("Expert Hidden", "mean"),
    "Signal-expert_hidden_scores-max": ("Expert Hidden", "max"),
    "Signal-expert_similarities-mean": ("Expert Similarity", "mean"),
    "Signal-expert_similarities-max": ("Expert Similarity", "max"),
    "Signal-expert_usage_entropy-mean": ("Usage Entropy", "mean"),
    "Signal-expert_usage_entropy-max": ("Usage Entropy", "max"),
    "Signal-expert_usage_gini-mean": ("Usage Gini", "mean"),
    "Signal-expert_usage_gini-max": ("Usage Gini", "max"),
    "Signal-expert_usage_effective_experts-mean": ("Inv. Herfindahl", "mean"),
    "Signal-expert_usage_effective_experts-max": ("Inv. Herfindahl", "max"),
    # Baselines
    "PredictiveEntropy-mean": ("Logit Entropy", "mean"),
    "PredictiveEntropy-max": ("Logit Entropy", "max"),
    "LLM-Check-attention-mean": ("LLM-Check Att.", "mean"),
    "LLM-Check-attention-max": ("LLM-Check Att.", "max"),
    "LLM-Check-hidden-mean": ("LLM-Check Hid.", "mean"),
    "LLM-Check-hidden-max": ("LLM-Check Hid.", "max"),
    "LLM-Check-entropy-mean": ("LLM-Check Entropy", "mean"),
    "LLM-Check-entropy-max": ("LLM-Check Entropy", "max"),
    "LLM-Check-perplexity": ("Perplexity", "none"),
    "HaluNet": ("HaluNet", "none"),
    "SemanticUncertainty": ("Semantic Uncertainty", "none"),
    "SemanticEnergy": ("Semantic Energy", "none"),
    "SelfCheckGPT-NLI": ("SelfCheckGPT (NLI)", "none"),
    "SelfCheckGPT-PROMPT": ("SelfCheckGPT (Prompt)", "none"),
    # Detector (for reference)
    "Ours-XGBoost (max)": ("InnerExpert (XGBoost)", "max"),
    "Ours-XGBoost (mean)": ("InnerExpert (XGBoost)", "mean"),
}

# Token-level: raw method -> display_name (no aggregation)
SIGNAL_METHOD_NAMES_TOKEN = {
    "Signal-router_entropy": "Router Entropy",
    "Signal-expert_hidden_scores": "Expert Hidden",
    "Signal-expert_similarities": "Expert Similarity",
    "Signal-expert_usage_entropy": "Usage Entropy",
    "Signal-expert_usage_gini": "Usage Gini",
    "Signal-expert_usage_effective_experts": "Inv. Herfindahl",
    "PredictiveEntropy": "Logit Entropy",
    "LLM-Check-attention": "LLM-Check Att.",
    "LLM-Check-hidden": "LLM-Check Hid.",
    "LLM-Check-entropy": "LLM-Check Entropy",
    "Ours-XGBoost": "InnerExpert (XGBoost)",
}


def _load_comparison_per_model(model_slug: str, level: str, metric_col: str = "auroc") -> pd.DataFrame:
    """Load and combine OOS + RTQA comparison tables for one model."""
    df_oos = load_oos_comparison(model_slug, level=level)
    if level == "answer":
        df_rtqa = load_rtqa_analysis(model_slug, level=level)
        df_rtqa["dataset"] = "realtimeqa-2026"
        df_combined = pd.concat([df_oos, df_rtqa], ignore_index=True)
    else:
        # Token-level RTQA: single-dataset table without 'dataset' column
        df_rtqa = load_rtqa_analysis(model_slug, level=level)
        df_rtqa["dataset"] = "realtimeqa-2026"
        df_combined = pd.concat([df_oos, df_rtqa], ignore_index=True)
    return df_combined[["dataset", "method", metric_col]]


def generate_signal_contribution_table(metric="AUROC") -> pd.DataFrame:
    """Produce a LaTeX table showing answer-level and token-level AUROC for
    each individual MoE signal and baseline method.

    Each AUROC value is averaged across all 5 evaluation datasets per host
    model. For answer-level scores, the best aggregation (mean or max) is
    selected per method based on the highest average AUROC across both
    models; the chosen aggregation is reported in the caption.
    """
    model_names = list(MODELS.values())

    # Per-dataset AUROC data, kept for ranking
    # answer_per_ds[model_name] = DataFrame[base_name × dataset] using best agg
    # token_per_ds[model_name]  = DataFrame[base_name × dataset]
    answer_per_ds = {}
    token_per_ds = {}

    # Per-model averaged AUROC (for display columns)
    model_answer_avg = {}
    model_token_avg = {}

    for model_slug, model_name in MODELS.items():
        # --- Answer-level ---
        df_ans = _load_comparison_per_model(model_slug, "answer", METRICS_INV[metric])
        df_ans = df_ans[df_ans["method"].isin(SIGNAL_METHOD_NAMES_ANSWER)].copy()
        df_ans[["base_name", "agg"]] = pd.DataFrame(
            df_ans["method"].map(SIGNAL_METHOD_NAMES_ANSWER).tolist(),
            index=df_ans.index,
        )

        # Pivot to (base_name, agg) × dataset -> auroc
        ans_pivot = df_ans.pivot_table(
            index=["base_name", "agg"],
            columns="dataset",
            values=METRICS_INV[metric],
            aggfunc="mean",
        )

        # Unfold "none" into both mean and max (fill where NaN, add new rows)
        if "none" in ans_pivot.index.get_level_values("agg"):
            none_rows = ans_pivot.xs("none", level="agg")
            ans_pivot = ans_pivot.drop(index="none", level="agg")
            for agg in ["mean", "max"]:
                if agg in ans_pivot.index.get_level_values("agg"):
                    # Fill NaN in existing agg rows with none values
                    existing = ans_pivot.xs(agg, level="agg")
                    mask = existing.isna() & none_rows.notna()
                    existing = existing.where(~mask, none_rows)
                    ans_pivot.loc[(slice(None), agg), :] = existing.values
                # Add none-method rows that don't exist yet for this agg
                missing = none_rows.index.difference(
                    ans_pivot.xs(agg, level="agg").index
                    if agg in ans_pivot.index.get_level_values("agg")
                    else pd.Index([])
                )
                if len(missing) > 0:
                    new_rows = none_rows.loc[missing].copy()
                    new_rows["agg"] = agg
                    new_rows = new_rows.set_index("agg", append=True)
                    ans_pivot = pd.concat([ans_pivot, new_rows])

        # Per-model average across datasets: (base_name, agg) -> mean auroc
        model_answer_avg[model_name] = ans_pivot.mean(axis=1).unstack("agg")

        # --- Token-level ---
        df_tok = _load_comparison_per_model(model_slug, "token", METRICS_INV[metric])
        df_tok = df_tok[df_tok["method"].isin(SIGNAL_METHOD_NAMES_TOKEN)].copy()
        df_tok["base_name"] = df_tok["method"].map(SIGNAL_METHOD_NAMES_TOKEN)

        tok_pivot = df_tok.pivot_table(
            index="base_name",
            columns="dataset",
            values=METRICS_INV[metric],
            aggfunc="mean",
        )
        model_token_avg[model_name] = tok_pivot.mean(axis=1)

        # Store per-dataset pivots for ranking (filled later with best agg)
        answer_per_ds[model_name] = ans_pivot  # base_name × dataset × agg
        token_per_ds[model_name] = tok_pivot  # base_name × dataset

    # --- Select best aggregation per method ---
    all_methods = set()
    for df in model_answer_avg.values():
        all_methods.update(df.index)

    best_agg_map = {}  # base_name -> "mean" or "max"
    for method in all_methods:
        scores = {}
        for agg in ["mean", "max"]:
            vals = []
            for mn in model_names:
                df = model_answer_avg.get(mn)
                if df is not None and method in df.index and agg in df.columns:
                    vals.append(df.loc[method, agg])
            if vals:
                scores[agg] = np.nanmean(vals)
        if not scores:
            continue
        best_agg_map[method] = max(scores, key=scores.get)

    # --- Build per-dataset answer DataFrames using best aggregation ---
    answer_ds_best = {}  # model_name -> DataFrame[base_name × dataset]
    for mn in model_names:
        ans_pivot = answer_per_ds.get(mn)
        if ans_pivot is None:
            continue
        rows = {}
        for method in sorted(all_methods):
            agg = best_agg_map.get(method)
            if agg is None:
                continue
            try:
                row = ans_pivot.loc[(method, agg)]
            except KeyError:
                row = pd.Series(np.nan, index=ans_pivot.columns)
            rows[method] = row
        answer_ds_best[mn] = pd.DataFrame(rows).T

    # --- Build final table ---
    rows = []
    for method in sorted(all_methods):
        agg = best_agg_map.get(method)
        if agg is None:
            continue
        row = {"Method": method}
        for mn in model_names:
            # Answer (averaged across datasets for display)
            df = model_answer_avg.get(mn)
            if df is not None and method in df.index and agg in df.columns:
                row[f"{mn} Answer"] = df.loc[method, agg]
            else:
                row[f"{mn} Answer"] = np.nan
            # Token (averaged across datasets for display)
            tok = model_token_avg.get(mn)
            if tok is not None and method in tok.index:
                row[f"{mn} Token"] = tok[method]
            else:
                row[f"{mn} Token"] = np.nan
        rows.append(row)

    df_final = pd.DataFrame(rows).set_index("Method")

    # Sort by Answer AUROC averaged across both models, descending
    answer_cols = [f"{mn} Answer" for mn in model_names]
    sort_key = df_final[answer_cols].mean(axis=1)
    df_final = df_final.loc[sort_key.sort_values(ascending=False).index]

    # Round and fill missing token values
    df_export = df_final.copy()
    for mn in model_names:
        df_export[f"{mn} Answer"] = df_export[f"{mn} Answer"].round(3)
        token_col = f"{mn} Token"
        df_export[token_col] = df_export[token_col].round(3)
        token_mask = df_export[token_col].isna()
        if token_mask.any():
            df_export[token_col] = df_export[token_col].astype(object)
            df_export.loc[token_mask, token_col] = "---"

    # Build two-level column header
    col_tuples = []
    for mn in model_names:
        col_tuples.append((mn, "Answer"))
        col_tuples.append((mn, "Token"))
    # col_tuples.append(("Avg. Rank", "Answer"))
    # col_tuples.append(("Avg. Rank", "Token"))
    df_export.columns = pd.MultiIndex.from_tuples(col_tuples, names=["Model", "Score"])
    df_export.index.name = "Method"

    # Build caption with aggregation groupings
    mean_methods = sorted([m for m, a in best_agg_map.items() if a == "mean"])
    max_methods = sorted([m for m, a in best_agg_map.items() if a == "max"])
    caption = (
        f"Average {metric} for individual MoE signals, baseline methods, and "
        "the best detector (InnerExpert XGBoost). Each value is averaged "
        "across all five evaluation datasets per host model. Answer-level "
        f"scores use the best aggregation per method (determined by highest "
        f"average {metric} across both models). "
        f"Aggregations used: max ({', '.join(max_methods)}); "
        f"mean ({', '.join(mean_methods)}). "
        f"``Token'' is the direct token-level {metric}. Methods without a "
        "token-level counterpart are marked ``---''."
    )

    df_export.to_latex(
        TABLES_DIR / f"signal_contribution_{metric}.tex",
        index=True,
        escape=False,
        float_format="%.3f",
        caption=caption,
        label=f"tab:signal_contribution_{metric}",
        column_format="l" + "c" * len(df_export.columns),
    )
    print(f"  -> {TABLES_DIR / f'signal_contribution_{metric}.tex'}")
    return df_final


# ---------------------------------------------------------------------------
# Human validation of LLM-as-a-judge labeling
# ---------------------------------------------------------------------------
def load_human_validation() -> pd.DataFrame:
    """Load and clean the annotated human validation spreadsheet."""
    if not HUMAN_VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Human validation file not found: {HUMAN_VALIDATION_PATH}")
    df = pd.read_excel(HUMAN_VALIDATION_PATH)
    df["label_weak_hallucination"] = df["label_weak_hallucination"].astype(int)
    df["label_llm_answer"] = df["label_llm_answer"].astype(int)
    df["human_label"] = pd.to_numeric(df["human_label"], errors="coerce").astype(int)
    # Map display names back to slugs for consistent filtering
    name_to_slug = {v: k for k, v in MODELS.items()}
    df["model_slug"] = df["model"].map(name_to_slug).fillna(df["model"])
    return df


def generate_judge_validation_tables() -> None:
    """Produce LaTeX confusion-matrix tables for the human validation of
    LLM-as-a-judge and weak heuristic labels.

    Tables (rows = human, columns = labeler):
    - ``judge_validation_confusion_llm.tex`` — Human vs. LLM-as-a-judge.
    - ``judge_validation_confusion_weak.tex`` — Human vs. weak heuristic.
    """
    df = load_human_validation()

    label_names = ["Grounded", "Hallucinated"]

    for labeler_name, col, filename, lbl in [
        ("LLM-as-a-judge", "label_llm_answer",
         "judge_validation_confusion_llm.tex", "tab:judge_validation_confusion_llm"),
        ("Weak heuristic", "label_weak_hallucination",
         "judge_validation_confusion_weak.tex", "tab:judge_validation_confusion_weak"),
    ]:
        cm = confusion_matrix(df["human_label"], df[col], labels=[0, 1])
        df_cm = pd.DataFrame(
            cm,
            index=pd.Index(label_names, name="Human"),
            columns=pd.Index(label_names, name=labeler_name),
        )

        df_cm.to_latex(
            TABLES_DIR / filename,
            index=True,
            caption=(
                f"Confusion matrix between human annotations and {labeler_name} "
                f"labels on the 200-sample validation set "
                f"(pooled across both host models, 25 per $2{{\\times}}2$ "
                f"contingency cell per model)."
            ),
            label=lbl,
            column_format="l" + "c" * len(df_cm.columns),
        )

# ---------------------------------------------------------------------------
# Inference time & memory benchmark
# ---------------------------------------------------------------------------

# Directory where 7.2-time-inference-analysis.py saves its CSV output
BENCHMARK_DIR = DATA_DIR / "inference-benchmark"


# Row order for the inference benchmark table (matches the method order in 7.2)
BENCHMARK_METHOD_ORDER = [
    "Vanilla",
    "Logit Entropy",
    "Perplexity",
    "LLM-Check (att.)",
    "LLM-Check (hid.)",
    "HaluNet",
    "Router Entropy",
    "Expert Hidden",
    "Expert Similarity",
    "Usage Entropy",
    "Usage Gini",
    "Inv. Herfindahl",
    "InnerExpert (LR)",
    "InnerExpert (MLP)",
    "InnerExpert (RF)",
    "InnerExpert (XGBoost)",
    "InnerExpert (Transformer)",
    "Semantic Uncertainty",
    "Semantic Energy",
    "SelfCheckGPT (NLI)",
    "SelfCheckGPT (Prompt)",
]


def generate_inference_benchmark_table() -> Optional[pd.DataFrame]:
    """Produce the inference time & memory benchmark LaTeX table.

    Reads ``data/inference-benchmark/{model_slug}/benchmark.csv`` produced by
    ``7.2-time-inference-analysis.py`` for both host models.

    Layout: rows = methods, columns = (Model, [Time / 100 tok, Peak GPU GB]).
    Methods are sorted in the order defined by ``BENCHMARK_METHOD_ORDER``.
    """
    frames = []
    for model_slug, model_name in MODELS.items():
        csv_path = BENCHMARK_DIR / model_slug / "benchmark.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        df_small = pd.read_csv(csv_path)
        df_small = df_small[["method", "time_per_100_tokens_s", "peak_gpu_mem_gb"]]
        df_small.columns = ["Method", f"time_{model_name}", f"mem_{model_name}"]
        frames.append(df_small)

    if not frames:
        print("  No benchmark CSVs found, skipping inference benchmark table.")
        return None

    # Merge on Method (each method appears once per model)
    from functools import reduce
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on="Method", how="outer"),
        frames,
    )

    # Sort by BENCHMARK_METHOD_ORDER
    order_map = {name: i for i, name in enumerate(BENCHMARK_METHOD_ORDER)}
    df_merged["_sort"] = df_merged["Method"].map(lambda x: order_map.get(x, 999))
    df_merged = df_merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    # Round values
    model_names = list(MODELS.values())
    for mn in model_names:
        df_merged[f"time_{mn}"] = df_merged[f"time_{mn}"].round(4)
        df_merged[f"mem_{mn}"] = df_merged[f"mem_{mn}"].round(2)

    # Build MultiIndex columns: (Model, metric)
    col_tuples = []
    for mn in model_names:
        col_tuples.append((mn, "Time / 100 tok (s)"))
        col_tuples.append((mn, "Peak GPU (GB)"))

    df_export = df_merged.copy()
    ordered_cols = []
    for mn in model_names:
        ordered_cols.append(df_export[f"time_{mn}"])
        ordered_cols.append(df_export[f"mem_{mn}"])
    df_export = pd.DataFrame(
        {ct: vals.values for ct, vals in zip(col_tuples, ordered_cols)}
    )
    df_export.index = df_merged["Method"]
    df_export.columns = pd.MultiIndex.from_tuples(col_tuples, names=["Model", ""])
    df_export.index.name = "Method"

    df_export.to_latex(
        TABLES_DIR / "inference_benchmark.tex",
        index=True,
        escape=False,
        float_format="%.3f",
        caption=(
            "Inference time (per 100 displayed tokens) and peak GPU memory for "
            "each detection method, measured on 50 RealTimeQA questions "
            "(June 2026) per host model. Both models are 4-bit quantized.  "
            "``Time / 100 tok'' is end-to-end wall-clock per 100 generated "
            "tokens (generation + scoring); for sampling-based methods it "
            "includes the cost of all background samples but counts only the "
            "displayed answer in the token denominator. ``Peak GPU (GB)' is "
            "the maximum GPU memory allocated during execution.  "
            "SelfCheckGPT (Prompt) uses Llama-2-7b-chat-hf as the scoring LLM "
            "(loaded on a second GPU for the Gemma-4-26B host).  "
            "InnerExpert times include MoE feature extraction "
            "(SVD-based hidden scores, attention scores, expert routing signals) "
            "as well as classifier inference."
        ),
        label="tab:inference_benchmark",
        column_format="l" + "c" * len(df_export.columns),
    )
    print(f"  -> {TABLES_DIR / 'inference_benchmark.tex'}")
    return df_export


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    return df_answers_table, df_tokens_table


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("8.0 — RESULTS COMPILATION FOR PAPER")
    print("=" * 70)

    # Load answer-level results and generate tables
    df_answers = compile_results(level="answer", drop_unmapped=True)
    df_tokens = compile_results(level="token", drop_unmapped=True)

    # Dataset statistics table
    print("\nGenerating dataset statistics table...")
    generate_dataset_statistics_table()


    for metric in ["AUROC", "F1"]:
        df_answers_table = results_table(df_answers, metric=metric)

        # Export table
        format_results_table_for_export(df_answers_table).to_latex(
            TABLES_DIR / f"answer_level_results_table_{metric}.tex",
            index=True,
            caption=f"Answer-level results ({metric}) across datasets and host models.",
            label="tab:answer_level_results",
            # float_format="%.3f",
            column_format="ll" + "c" * (len(df_answers_table.columns)),
        )

        # Load token-level results and generate tables
        df_tokens_table = results_table(df_tokens, metric=metric)

        # Export table
        format_results_table_for_export(df_tokens_table).to_latex(
            TABLES_DIR / f"token_level_results_table_{metric}.tex",
            index=True,
            caption=f"Token-level results ({metric}) across datasets and host models.",
            label="tab:token_level_results",
            # float_format="%.3f",
            column_format="ll" + "c" * (len(df_tokens_table.columns)),
        )

        # Signal contribution analysis
        print("\nGenerating signal contribution table...")
        generate_signal_contribution_table(metric=metric)

    # Plot roc curves

    # Discriminatory power plots for each signal method


    # Analysis of LLM-as-a-judge approach with human annotated sample
    print("\nGenerating judge validation tables...")
    generate_judge_validation_tables()

    # Inference time & memory benchmark table
    print("\nGenerating inference benchmark table...")
    generate_inference_benchmark_table()
