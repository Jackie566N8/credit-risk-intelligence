import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from model_display_config import METRIC_LABELS


def make_metric_bar_chart(df: pd.DataFrame, metric: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = df.sort_values(metric, ascending=False)
    sns.barplot(data=plot_df, x="model_label", y=metric, ax=ax, color="#4C78A8")
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Model")
    ax.set_xlabel("")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(1.0, float(plot_df[metric].max()) * 1.08))
    fig.tight_layout()
    return fig
