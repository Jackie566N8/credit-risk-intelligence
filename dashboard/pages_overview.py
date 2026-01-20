import streamlit as st

from dashboard.charts import (
    make_auc_comparison_chart,
    make_metric_bar_chart,
    make_precision_recall_chart,
)
from dashboard.components import metric_cards, numeric_table, page_header
from dashboard.config import METRIC_LABELS


def render_overview_page(comparison_df) -> None:
    page_header(
        "Portfolio Overview",
        "Model performance comparison and optimized-threshold classification metrics.",
    )
    metric_cards(comparison_df)

    if comparison_df.empty:
        return

    display_columns = [
        "model_label",
        "cv_roc_auc_mean",
        "test_roc_auc",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "accuracy",
    ]
    numeric_table(comparison_df[display_columns])

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_metric = st.selectbox(
            "Metric",
            options=["test_roc_auc", "recall", "precision", "f1", "f2", "accuracy"],
            format_func=lambda value: METRIC_LABELS.get(value, value),
        )
        st.pyplot(make_metric_bar_chart(comparison_df, selected_metric))
    with col2:
        st.pyplot(make_auc_comparison_chart(comparison_df))

    st.pyplot(make_precision_recall_chart(comparison_df))
