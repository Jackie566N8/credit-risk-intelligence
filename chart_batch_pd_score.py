import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_batch_pd_score_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=df,
        x="credit_score",
        y="probability_default",
        hue="decision",
        hue_order=["Approve", "Manual Review", "Decline"],
        ax=ax,
        s=70,
    )
    ax.set_title("Credit Score vs Default Probability")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Default Probability")
    ax.set_ylim(0, max(0.6, float(df["probability_default"].max()) * 1.1))
    ax.legend(title="")
    fig.tight_layout()
    return fig
