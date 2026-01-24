import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_batch_decision_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="decision", order=["Approve", "Manual Review", "Decline"], ax=ax, color="#4C78A8")
    ax.set_title("Batch Decision Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Applications")
    fig.tight_layout()
    return fig
