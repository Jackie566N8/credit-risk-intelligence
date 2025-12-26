import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dashboard.config import METRIC_LABELS


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


def make_precision_recall_chart(df: pd.DataFrame):
    value_columns = ["precision", "recall", "f1", "f2"]
    plot_df = df.melt(
        id_vars=["model_label"],
        value_vars=value_columns,
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


def make_auc_comparison_chart(df: pd.DataFrame):
    plot_df = df.melt(
        id_vars=["model_label"],
        value_vars=["cv_roc_auc_mean", "test_roc_auc"],
        var_name="metric",
        value_name="roc_auc",
    )
    plot_df["metric"] = plot_df["metric"].map(METRIC_LABELS).fillna(plot_df["metric"])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.lineplot(
        data=plot_df,
        x="model_label",
        y="roc_auc",
        hue="metric",
        marker="o",
        ax=ax,
    )
    ax.set_title("Cross-Validation vs Holdout ROC-AUC")
    ax.set_xlabel("")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    fig.tight_layout()
    return fig


def make_feature_importance_chart(df: pd.DataFrame, model_label: str, top_n: int = 15):
    plot_df = df.head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="importance", y="feature", ax=ax, color="#54A24B")
    ax.set_title(f"{model_label} Top Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


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
