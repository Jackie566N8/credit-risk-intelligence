from io import StringIO

import pandas as pd
import streamlit as st

from charts.chart_batch_decision_count import make_batch_decision_chart
from charts.chart_batch_pd_score import make_batch_pd_score_chart
from config.credit_decision_config import BASE_LOSS_GIVEN_DEFAULT
from config.page_step_config import PAGE_STEPS
from risk.risk_applicant_profile import REQUIRED_BATCH_COLUMNS
from risk.risk_batch_assessment import assess_batch
from risk.risk_batch_template import sample_batch_template
from ui.ui_data_table import numeric_table
from ui.ui_operation_steps import operation_steps
from ui.ui_page_header import page_header


def render_batch_page() -> None:
    page_header(
        "Batch Risk Assessment",
        "CSV batch scoring with approval decision, credit score, and expected loss output.",
    )
    operation_steps("Batch Risk Assessment", PAGE_STEPS["Batch Risk Assessment"])

    lgd = st.slider(
        "Batch Loss Given Default",
        min_value=0.20,
        max_value=0.90,
        value=BASE_LOSS_GIVEN_DEFAULT,
        step=0.05,
        help="Used to calculate batch expected loss. Expected loss = loan amount x default probability x LGD.",
    )
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload batch applicant data. Field names must match Required columns.",
    )
    if uploaded_file:
        with st.spinner("Reading uploaded CSV file..."):
            input_df = pd.read_csv(uploaded_file)
    else:
        input_df = sample_batch_template()
    if uploaded_file is None:
        st.caption("Using sample applications.")
    elif len(input_df) > 5000:
        st.info(f"Large batch detected: {len(input_df):,} applications. Scoring may take a short time.")

    with st.expander("Required columns", expanded=False):
        st.code(", ".join(REQUIRED_BATCH_COLUMNS), language="text")

    with st.spinner(f"Scoring {len(input_df):,} applications and calculating expected loss..."):
        try:
            result_df = assess_batch(input_df, lgd=lgd)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    batch_summary(result_df)
    st.caption(_probability_source_caption(result_df))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(make_batch_decision_chart(result_df))
    with col2:
        st.pyplot(make_batch_pd_score_chart(result_df))

    numeric_table(result_df)
    st.download_button(
        "Download Risk Assessment CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_risk_assessment_results.csv",
        mime="text/csv",
        help="Download scored results with default probability, credit score, approval recommendation, and expected loss.",
    )

    template_csv = sample_batch_template().to_csv(index=False)
    st.download_button(
        "Download CSV Template",
        data=StringIO(template_csv).getvalue().encode("utf-8"),
        file_name="batch_risk_assessment_template.csv",
        mime="text/csv",
        help="Download the batch assessment template. Fill it in and upload it directly.",
    )


def batch_summary(result_df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Applications", f"{len(result_df):,}")
    col2.metric("Avg Default Probability", f"{result_df['probability_default'].mean():.2%}")
    col3.metric("Avg Credit Score", f"{result_df['credit_score'].mean():.0f}")
    col4.metric("Total Expected Loss", f"${result_df['expected_loss'].sum():,.0f}")


def _probability_source_caption(result_df: pd.DataFrame) -> str:
    sources = result_df["probability_source"].value_counts().to_dict()
    return "Probability source: " + ", ".join(f"{name}={count}" for name, count in sources.items())
