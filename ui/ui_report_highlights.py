import streamlit as st

from config.report_metrics import REPORT_RANDOM_FOREST_METRICS


def report_highlight_cards() -> None:
    st.markdown("### Project Report Highlight")
    col1, col2, col3 = st.columns(3)
    col1.metric("Report Model", REPORT_RANDOM_FOREST_METRICS["model"])
    col2.metric("Report ROC-AUC", f"{REPORT_RANDOM_FOREST_METRICS['roc_auc']:.2f}")
    col3.metric("Report Default Recall", f"{REPORT_RANDOM_FOREST_METRICS['default_recall']:.2f}")
    st.caption(
        "Main risk factors identified in the report: "
        f"{REPORT_RANDOM_FOREST_METRICS['main_risk_factors']}."
    )
