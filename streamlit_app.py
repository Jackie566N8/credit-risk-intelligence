import streamlit as st

from dashboard.components import show_result_status
from dashboard.config import MODEL_RESULT_FILES, PAGE_LABELS
from dashboard.data_loader import load_all_model_reports, load_comparison_results
from dashboard.pages_batch import render_batch_page
from dashboard.pages_decision import render_decision_page
from dashboard.pages_figures import render_figures_page
from dashboard.pages_model_detail import render_model_detail_page
from dashboard.pages_overview import render_overview_page
from dashboard.styles import apply_app_style


st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_comparison_results():
    return load_comparison_results()


@st.cache_data(show_spinner=False)
def cached_model_reports():
    return load_all_model_reports()


def main() -> None:
    apply_app_style()
    st.title("Credit Risk Intelligence Dashboard")
    st.caption("LendingClub risk modeling, credit scoring, approval decisioning, and portfolio visualization.")

    comparison_df = cached_comparison_results()
    reports = cached_model_reports()
    show_result_status(comparison_df, list(MODEL_RESULT_FILES.values()))

    page = st.sidebar.radio(
        "View",
        options=list(PAGE_LABELS),
        format_func=lambda value: PAGE_LABELS[value],
    )

    if page == "Portfolio Overview":
        render_overview_page(comparison_df)
    elif page == "Model Detail":
        default_model = None
        if not comparison_df.empty:
            default_model = comparison_df.sort_values("test_roc_auc", ascending=False).iloc[0]["model"]
        render_model_detail_page(reports, default_model=default_model)
    elif page == "Single Customer Decision":
        render_decision_page()
    elif page == "Batch Risk Assessment":
        render_batch_page()
    elif page == "EDA Figures":
        render_figures_page()


if __name__ == "__main__":
    main()
