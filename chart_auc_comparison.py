import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from model_display_config import METRIC_LABELS


def make_auc_comparison_chart(df: pd.DataFrame):
    plot_df = df.melt(
        id_vars=["model_label"],
        value_vars=["cv_roc_auc_mean", "test_roc_auc"],
        var_name="metric",
        value_name="roc_auc",
    )
    plot_df["metric"] = plot_df["metric"].map(METRIC_LABELS).fillna(plot_df["metric"])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.lineplot(data=plot_df, x="model_label", y="roc_auc", hue="metric", marker="o", ax=ax)
    ax.set_title("Cross-Validation vs Holdout ROC-AUC")
    ax.set_xlabel("")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    return fig
