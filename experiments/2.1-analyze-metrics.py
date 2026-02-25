"""
This script analyzes the metrics collected from the experiments and generates
visualizations to assess their separability and whether it is feasible to use
them as ground-truth labels. 

It reads the metrics from a specified directory, processes the data, and
creates plots to illustrate the results.
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


model_slug = "allenai__OLMoE-1B-7B-0924-Instruct"
dataset_slug = "realtimeqa-2026-01"

data_dir = Path("data") / model_slug / dataset_slug
figures_dir = Path("figures") / "2.1-analyze-metrics"
figures_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(data_dir / "results.parquet")

SCORE_COLS = df.columns[
    df.columns.map(
        lambda x: any(
            x.startswith(pref) 
            for pref in ["rouge", "bert", "bleu"]
        )
    )
].tolist()

df_metrics = df[SCORE_COLS].melt(var_name="metric", value_name="score")
df_metrics["source"] = df_metrics["metric"].apply(
    lambda x: "evidence-based" if x.endswith("rag") else "base"
)
df_metrics["metric"] = df_metrics["metric"].apply(
    lambda x: x.replace("_rag", "")
)

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
sns.violinplot(x="metric", y="score", hue="source", data=df_metrics, split=True, density_norm="count")
plt.title("Distribution of Metrics by Source")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.legend(title="Source")
plt.tight_layout()
plt.savefig(figures_dir / "metrics_violin_plot.png")
