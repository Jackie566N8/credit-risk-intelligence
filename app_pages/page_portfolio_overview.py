import streamlit as st

from charts.chart_auc_comparison import make_auc_comparison_chart
from charts.chart_metric_bar import make_metric_bar_chart
from charts.chart_precision_recall import make_precision_recall_chart
from config.model_display_config import METRIC_LABELS
from config.page_step_config import PAGE_STEPS
from ui.ui_data_table import numeric_table
from ui.ui_metric_cards import metric_cards
from ui.ui_operation_steps import operation_steps
from ui.ui_page_header import page_header
from ui.ui_report_highlights import report_highlight_cards


def render_overview_page(comparison_df) -> None:
    page_header(
        "Portfolio Overview",
        "Model performance comparison and optimized-threshold classification metrics.",
    )
    operation_steps("Portfolio Overview", PAGE_STEPS["Portfolio Overview"])
    report_highlight_cards()
    st.markdown("### Reproducible Training Results")
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
            help="Choose a metric. The bar chart ranks models by the selected metric.",
        )
        st.pyplot(make_metric_bar_chart(comparison_df, selected_metric))
    with col2:
        st.pyplot(make_auc_comparison_chart(comparison_df))

    st.pyplot(make_precision_recall_chart(comparison_df))
