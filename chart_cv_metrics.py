import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_cv_metric_chart(cv_df: pd.DataFrame, model_label: str):
    if cv_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=cv_df, x="metric", y="mean", ax=ax, color="#72B7B2")
    ax.errorbar(
        x=range(len(cv_df)),
        y=cv_df["mean"],
        yerr=cv_df["std"],
        fmt="none",
        ecolor="#333333",
        capsize=4,
    )
    ax.set_title(f"{model_label} Cross-Validation Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
