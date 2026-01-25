import pandas as pd
import streamlit as st


def metric_cards(df: pd.DataFrame) -> None:
    if df.empty:
        return

    best_auc = df.sort_values("test_roc_auc", ascending=False).iloc[0]
    best_recall = df.sort_values("recall", ascending=False).iloc[0]
    best_f2 = df.sort_values("f2", ascending=False).iloc[0]
    lowest_threshold = df.sort_values("threshold", ascending=True).iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Test ROC-AUC", best_auc["model_label"], f"{best_auc['test_roc_auc']:.4f}")
    col2.metric("Highest Recall", best_recall["model_label"], f"{best_recall['recall']:.4f}")
    col3.metric("Best F2", best_f2["model_label"], f"{best_f2['f2']:.4f}")
    col4.metric("Lowest Threshold", lowest_threshold["model_label"], f"{lowest_threshold['threshold']:.2f}")
