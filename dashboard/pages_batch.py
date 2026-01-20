from io import StringIO

import pandas as pd
import streamlit as st

from dashboard.charts import make_batch_decision_chart, make_batch_pd_score_chart
from dashboard.components import numeric_table, page_header
from dashboard.config import BASE_LOSS_GIVEN_DEFAULT
from dashboard.risk_engine import REQUIRED_BATCH_COLUMNS, assess_batch, sample_batch_template


def render_batch_page() -> None:
    page_header(
        "Batch Risk Assessment",
        "CSV batch scoring with approval decision, credit score, and expected loss output.",
    )

    lgd = st.slider("Batch Loss Given Default", min_value=0.20, max_value=0.90, value=BASE_LOSS_GIVEN_DEFAULT, step=0.05)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        input_df = sample_batch_template()
        st.caption("Using sample applications.")
    else:
        input_df = pd.read_csv(uploaded_file)

    with st.expander("Required columns", expanded=False):
        st.code(", ".join(REQUIRED_BATCH_COLUMNS), language="text")

    try:
        result_df = assess_batch(input_df, lgd=lgd)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    _batch_summary(result_df)

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
    )

    template_csv = sample_batch_template().to_csv(index=False)
    st.download_button(
        "Download CSV Template",
        data=StringIO(template_csv).getvalue().encode("utf-8"),
        file_name="batch_risk_assessment_template.csv",
        mime="text/csv",
    )


def _batch_summary(result_df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Applications", f"{len(result_df):,}")
    col2.metric("Avg Default Probability", f"{result_df['probability_default'].mean():.2%}")
    col3.metric("Avg Credit Score", f"{result_df['credit_score'].mean():.0f}")
    col4.metric("Total Expected Loss", f"${result_df['expected_loss'].sum():,.0f}")
