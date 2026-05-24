import streamlit as st

from config.app_navigation_config import PAGE_LABELS
from config.app_paths import MODEL_ARTIFACT_FILES, MODEL_RESULT_FILES
from model_results.model_results_loader import load_all_model_reports, load_comparison_results
from pages.page_batch_risk_assessment import render_batch_page
from pages.page_eda_figures import render_figures_page
from pages.page_model_detail import render_model_detail_page
from pages.page_portfolio_overview import render_overview_page
from pages.page_single_customer_decision import render_decision_page
from ui.streamlit_page_style import apply_app_style
from ui.ui_result_status import show_result_status


st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner="Loading model comparison results...")
def cached_comparison_results():
    return load_comparison_results()


@st.cache_data(show_spinner="Loading model detail reports...")
def cached_model_reports():
    return load_all_model_reports()


def main() -> None:
    apply_app_style()
    st.title("Credit Risk Intelligence Dashboard")
    st.caption("LendingClub 2007-2018 risk modeling, credit scoring, approval decisioning, and portfolio visualization.")

    with st.spinner("Preparing dashboard data and model status..."):
        comparison_df = cached_comparison_results()
        reports = cached_model_reports()
    show_result_status(
        comparison_df,
        list(MODEL_RESULT_FILES.values()),
        list(MODEL_ARTIFACT_FILES.values()),
    )

    st.sidebar.markdown("### Project Modules")
    st.sidebar.caption("EDA, model comparison, customer decisioning, batch scoring, and expected loss.")

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
