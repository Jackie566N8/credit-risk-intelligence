import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config.model_display_config import METRIC_LABELS


def make_precision_recall_chart(df: pd.DataFrame):
    plot_df = df.melt(
        id_vars=["model_label"],
        value_vars=["precision", "recall", "f1", "f2"],
        var_name="metric",
        value_name="score",
    )
    plot_df["metric"] = plot_df["metric"].map(METRIC_LABELS).fillna(plot_df["metric"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=plot_df, x="model_label", y="score", hue="metric", ax=ax)
    ax.set_title("Optimized Threshold Classification Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="", ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    return fig
