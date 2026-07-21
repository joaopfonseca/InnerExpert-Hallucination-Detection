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

# Host models display names
MODELS = {
    "allenai__OLMoE-1B-7B-0924-Instruct": "OLMoE-1B-7B",
    "google__gemma-4-26B-A4B-it": "Gemma-4-26B",
}
MODEL_SLUGS = list(MODELS.keys())


# Method display names
METHOD_NAMES = {
    "Ours-XGBoost": "InnerExpert (XGBoost)",
    "Ours-LogisticRegression": "InnerExpert (LR)",
    "Ours-MLP": "InnerExpert (MLP)",
    "Ours-RandomForest": "InnerExpert (RF)",
    "Ours-Transformer": "InnerExpert (Transformer)",
    "Ours-XGBoost (mean)": "InnerExpert (XGBoost) (mean)",
    "Ours-LogisticRegression (mean)": "InnerExpert (LR) (mean)",
    "Ours-MLP (mean)": "InnerExpert (MLP) (mean)",
    "Ours-RandomForest (mean)": "InnerExpert (RF) (mean)",
    "Ours-Transformer (mean)": "InnerExpert (Transformer) (mean)",
    "Ours-XGBoost (max)": "InnerExpert (XGBoost) (max)",
    "Ours-LogisticRegression (max)": "InnerExpert (LR) (max)",
    "Ours-MLP (max)": "InnerExpert (MLP) (max)",
    "Ours-RandomForest (max)": "InnerExpert (RF) (max)",
    "Ours-Transformer (max)": "InnerExpert (Transformer) (max)",
    "HaluNet": "HaluNet",
    "SemanticUncertainty": "Semantic Uncertainty",
    "SemanticEnergy": "Semantic Energy",
    "SelfCheckGPT-NLI": "SelfCheckGPT (NLI)",
    "SelfCheckGPT-PROMPT": "SelfCheckGPT (Prompt)",
    "PredictiveEntropy-mean": "Logit Entropy (mean)",
    "PredictiveEntropy-max": "Logit Entropy (max)",
    "LLM-Check-attention-mean": "LLM-Check (Attention Score) (mean)",
    "LLM-Check-attention-max": "LLM-Check (Attention Score) (max)",
    "LLM-Check-hidden-mean": "LLM-Check (Hidden Score) (mean)",
    "LLM-Check-hidden-max": "LLM-Check (Hidden Score) (max)",
    "LLM-Check-entropy-mean": "DROP",  # "Entropy (LLM-Check) (mean)",
    "LLM-Check-entropy-max": "DROP",  # "Entropy (LLM-Check) (max)",
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


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
set_matplotlib_style(font_size=8, use_latex=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_oos_comparison(model_slug: str) -> pd.DataFrame:
    """Load the cross-dataset comparison CSV for a model."""
    path = DATA_DIR / "oos-comparison" / model_slug / "comparison_table_answer.csv"
    if not path.exists():
        raise FileNotFoundError(f"Comparison table not found: {path}")
    df = pd.read_csv(path)
    return df


def load_rtqa_analysis(model_slug: str) -> pd.DataFrame:
    """Load the RealtimeQA-2026 analysis results for a model."""
    path = RTQA_DIR / model_slug / "analysis" / "comparison_table_answer.csv"
    if not path.exists():
        raise FileNotFoundError(f"RTQA analysis not found: {path}")
    df = pd.read_csv(path)
    return df


def load_predictions(dataset: str, model_slug: str, method: str) -> pd.DataFrame:
    """Load prediction parquet for a specific dataset/model/method."""
    if dataset == "realtimeqa-2026":
        base = RTQA_DIR / model_slug / "predictions"
    else:
        base = DATA_DIR / f"oos-{dataset}" / model_slug / "predictions"
    path = base / f"{method}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    return pd.read_parquet(path)


def load_labeled(dataset: str, model_slug: str) -> pd.DataFrame:
    """Load labeled data for a dataset/model."""
    if dataset == "realtimeqa-2026":
        base = RTQA_DIR / model_slug
    else:
        base = DATA_DIR / f"oos-{dataset}" / model_slug
    path = base / "results_labeled_zai-org__GLM-5.1.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Labeled data not found: {path}")
    return pd.read_parquet(path)


def answer_level_results():
    # Answer-level results
    columns = ["dataset", "method"] + list(METRICS.keys())
    answer_dfs = []
    for model_slug, model_name in MODELS.items():
        print(f"\nProcessing model: {model_name} ({model_slug})")
        df_oos = load_oos_comparison(model_slug)
        df_rtqa = load_rtqa_analysis(model_slug)
        
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
        answer_dfs.append(df_combined)
    df_answers = pd.concat(answer_dfs, ignore_index=True)
    df_answers["dataset"] = df_answers["dataset"].map(DATASETS)
    df_answers["method"] = df_answers["method"].map(METHOD_NAMES)
    df_answers = df_answers[df_answers["method"] != "DROP"]
    df_answers = df_answers.sort_values(by=["dataset", "model", "method"])
    df_answers.rename(
        columns={
            "dataset": "Dataset", 
            "model": "Model",
            "method": "Method",
            **METRICS
        },
        inplace=True
    )
    return df_answers

def answer_level_results_table(df_answers: pd.DataFrame) -> pd.DataFrame:
    # Generate pivot table for F1 scores
    df_main = df_answers[
        ~df_answers["Method"].map(lambda x: x.endswith("mean)"))
    ]
    metric = "F1"
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
    df_main = df_main.round(3)
    df_main["Avg. Rank"] = ranks.values

    return df_main

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    return


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("8.0 — RESULTS COMPILATION FOR PAPER")
    print("=" * 70)

    df_answers = answer_level_results()
    df_answers_table = answer_level_results_table(df_answers)
