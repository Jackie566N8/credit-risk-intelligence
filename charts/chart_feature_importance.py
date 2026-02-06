import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_feature_importance_chart(df: pd.DataFrame, model_label: str, top_n: int = 15):
    plot_df = df.head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="importance", y="feature", ax=ax, color="#54A24B")
    ax.set_title(f"{model_label} Top Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig
