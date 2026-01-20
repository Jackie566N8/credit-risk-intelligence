import streamlit as st

from dashboard.components import assessment_metrics, decision_badge, driver_list, page_header
from dashboard.config import (
    BASE_LOSS_GIVEN_DEFAULT,
    HOME_OWNERSHIP_OPTIONS,
    PURPOSE_OPTIONS,
    TERM_OPTIONS,
    VERIFICATION_OPTIONS,
)
from dashboard.risk_engine import ApplicantProfile, assess_applicant


def render_decision_page() -> None:
    page_header(
        "Single Customer Decision",
        "Loan approval recommendation, 300-850 score, and expected loss estimate.",
    )

    with st.form("single_customer_form"):
        left, middle, right = st.columns(3)
        with left:
            loan_amnt = st.number_input("Loan Amount", min_value=500.0, max_value=40000.0, value=15000.0, step=500.0)
            annual_inc = st.number_input("Annual Income", min_value=1.0, max_value=500000.0, value=75000.0, step=1000.0)
            fico_score = st.slider("FICO Score", min_value=300, max_value=850, value=700)
        with middle:
            dti = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=60.0, value=18.0, step=0.5)
            revol_util = st.slider("Revolving Utilization", min_value=0.0, max_value=120.0, value=42.0, step=1.0)
            int_rate = st.slider("Interest Rate", min_value=5.0, max_value=31.0, value=13.5, step=0.1)
        with right:
            term_months = st.selectbox("Term", TERM_OPTIONS, index=0, format_func=lambda value: f"{value} months")
            purpose = st.selectbox("Purpose", PURPOSE_OPTIONS, index=0)
            home_ownership = st.selectbox("Home Ownership", HOME_OWNERSHIP_OPTIONS, index=0)
            verification_status = st.selectbox("Verification", VERIFICATION_OPTIONS, index=1)
            lgd = st.slider("Loss Given Default", min_value=0.20, max_value=0.90, value=BASE_LOSS_GIVEN_DEFAULT, step=0.05)

        submitted = st.form_submit_button("Run Assessment")

    if not submitted:
        st.info("Enter applicant attributes and run assessment.")
        return

    profile = ApplicantProfile(
        loan_amnt=loan_amnt,
        annual_inc=annual_inc,
        fico_score=fico_score,
        dti=dti,
        revol_util=revol_util,
        int_rate=int_rate,
        term_months=term_months,
        purpose=purpose,
        home_ownership=home_ownership,
        verification_status=verification_status,
    )
    assessment = assess_applicant(profile, lgd=lgd)

    decision_badge(assessment.decision)
    assessment_metrics(assessment)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Risk Drivers")
        driver_list(assessment.drivers)
    with col2:
        st.markdown("### Decision Details")
        st.write(f"Loss given default: {assessment.loss_given_default:.0%}")
        st.write(f"Estimated exposure: ${profile.loan_amnt:,.0f}")
        st.write(f"Loan-to-income ratio: {profile.loan_amnt / max(profile.annual_inc, 1):.1%}")
