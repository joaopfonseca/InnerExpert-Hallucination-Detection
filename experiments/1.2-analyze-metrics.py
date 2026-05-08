"""
This script analyzes the metrics collected from the experiments and generates
visualizations to assess their separability and whether it is feasible to use
them as ground-truth labels.

It reads the metrics from a specified directory, processes the data, and
creates plots to illustrate the results.
"""

import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    pass

from moeuncert.experiments import (
    optimal_threshold,
    resolve_model_slug,
    resolve_dataset_slug,
    build_figures_path,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze generation metrics for RealtimeQA experiments."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMoE-1B-7B-0924-Instruct",
        help="HuggingFace model name (default: allenai/OLMoE-1B-7B-0924-Instruct)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Year(s) for RealtimeQA, e.g. --years 2025 2026. Defaults to previous month year.",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        choices=range(1, 13),
        metavar="MONTH",
        help="Month (1-12) of the RealtimeQA dataset. "
             "Only used when a single year is provided. "
             "Defaults to the previous calendar month.",
    )
    args = parser.parse_args()

    model_slug = resolve_model_slug(args.model)
    years, month, dataset_slug = resolve_dataset_slug(args.years, args.month)

    data_dir = Path("data") / dataset_slug / model_slug
    figures_dir = build_figures_path("1.1-analyze-metrics", dataset_slug, model_slug)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_dir / "results.parquet")

    # Filter to only include rows where evidence is present
    df = df[df["evidence"].map(lambda x: len(x) > 0)]

    SCORE_COLS = df.columns[
        df.columns.map(
            lambda x: any(x.startswith(pref) for pref in ["rouge", "bert", "bleu"])
        )
    ].tolist()

    df_metrics = df[SCORE_COLS].melt(var_name="metric", value_name="score")
    df_metrics["source"] = df_metrics["metric"].apply(
        lambda x: "evidence-based" if x.endswith("rag") else "base"
    )
    df_metrics["metric"] = df_metrics["metric"].apply(lambda x: x.replace("_rag", ""))

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        x="metric",
        y="score",
        hue="source",
        data=df_metrics,
        split=True,
        density_norm="count",
    )
    plt.title("Distribution of Metrics by Source")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.legend(title="Source")
    plt.tight_layout()
    plt.savefig(figures_dir / "metrics_violin_plot.png")
    plt.close()

    #############################################################################
    # Binary classification setup
    # Label: RAG instances = 1 (correct), base instances = 0 (incorrect).
    # Each row contributes two samples to the stacked dataset.

    BASE_COLS = [c for c in SCORE_COLS if not c.endswith("_rag")]
    RAG_COLS = [c for c in SCORE_COLS if c.endswith("_rag")]

    base_df = df[BASE_COLS].copy()
    base_df["label"] = 0

    rag_df = df[RAG_COLS].rename(columns=dict(zip(RAG_COLS, BASE_COLS)))
    rag_df["label"] = 1

    stacked = pd.concat([base_df, rag_df], ignore_index=True)
    X = stacked[BASE_COLS].values
    y = stacked["label"].values

    #############################################################################
    # Logistic regression on all metrics simultaneously
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, y)
    lr_scores = lr.predict_proba(X_scaled)[:, 1]

    #############################################################################
    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 7))

    for col in BASE_COLS:
        fpr, tpr, _ = roc_curve(y, stacked[col].values)
        ax.plot(fpr, tpr, label=f"{col} (AUC={auc(fpr, tpr):.2f})")

    fpr_lr, tpr_lr, _ = roc_curve(y, lr_scores)
    ax.plot(
        fpr_lr,
        tpr_lr,
        linewidth=2.5,
        linestyle="--",
        label=f"logistic regression (AUC={auc(fpr_lr, tpr_lr):.2f})",
    )

    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, label="random")
    ax.set_title("ROC Curves — individual metrics and logistic regression")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close()

    #############################################################################
    # Confusion matrices at F1-optimal threshold
    classifiers = [(col, stacked[col].values) for col in BASE_COLS]
    classifiers.append(("logistic\nregression", lr_scores))

    n = len(classifiers)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))

    for ax, (name, scores) in zip(axes, classifiers):
        thresh, f1 = optimal_threshold(y, scores)
        y_pred = (scores >= thresh).astype(int)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(f"{name}\nthresh={thresh:.2f}  F1={f1:.2f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png", dpi=150)
    plt.close()

