from pathlib import Path

import pandas as pd
import streamlit as st


def show_result_status(comparison_df: pd.DataFrame, result_files: list[Path]) -> None:
    available_count = sum(path.exists() for path in result_files)
    st.caption(f"Loaded {available_count} model result files.")
    if comparison_df.empty:
        st.warning("No comparison result file found. Run `python train_credit_models.py` first.")


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


def numeric_table(df: pd.DataFrame) -> None:
    if df.empty:
        return

    display_df = df.copy()
    numeric_columns = display_df.select_dtypes(include=["number"]).columns
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column, format="%.4f")
            for column in numeric_columns
        },
    )


def show_raw_text(path: Path, text: str) -> None:
    with st.expander(f"Raw result file: {path.name}", expanded=False):
        st.code(text, language="text")
