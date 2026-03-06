import streamlit as st

from ui.ui_formatters import format_currency


def assessment_metrics(assessment) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Default Probability", f"{assessment.probability_default:.2%}")
    col2.metric("Credit Score", f"{assessment.credit_score}")
    col3.metric("Risk Grade", assessment.risk_grade)
    col4.metric("Expected Loss", format_currency(assessment.expected_loss))


def probability_source_panel(assessment) -> None:
    model_probability = (
        f"{assessment.model_probability_default:.2%}"
        if assessment.model_probability_default is not None
        else "Not available"
    )
    st.caption(
        " | ".join(
            [
                f"Probability source: {assessment.probability_source}",
                f"Model PD: {model_probability}",
                f"Scorecard PD: {assessment.scorecard_probability_default:.2%}",
            ]
        )
    )
