import streamlit as st

from config.credit_decision_config import (
    BASE_LOSS_GIVEN_DEFAULT,
    HOME_OWNERSHIP_OPTIONS,
    PURPOSE_OPTIONS,
    TERM_OPTIONS,
    VERIFICATION_OPTIONS,
)
from config.page_step_config import PAGE_STEPS
from risk.risk_applicant_profile import ApplicantProfile
from risk.risk_single_assessment import assess_applicant
from ui.ui_assessment_summary import assessment_metrics, probability_source_panel
from ui.ui_decision_badge import decision_badge
from ui.ui_operation_steps import operation_steps
from ui.ui_page_header import page_header
from ui.ui_risk_drivers import driver_list


def render_decision_page() -> None:
    page_header(
        "Single Customer Decision",
        "Loan approval recommendation, 300-850 score, and expected loss estimate.",
    )
    operation_steps("Single Customer Decision", PAGE_STEPS["Single Customer Decision"])

    with st.form("single_customer_form"):
        left, middle, right = st.columns(3)
        with left:
            loan_amnt = st.number_input(
                "Loan Amount",
                min_value=500.0,
                max_value=40000.0,
                value=15000.0,
                step=500.0,
                help="客户申请的贷款本金。金额越高，潜在风险敞口越大。",
            )
            annual_inc = st.number_input(
                "Annual Income",
                min_value=1.0,
                max_value=500000.0,
                value=75000.0,
                step=1000.0,
                help="客户年收入。系统会结合贷款金额计算贷款收入比。",
            )
            fico_score = st.slider(
                "FICO Score",
                min_value=300,
                max_value=850,
                value=700,
                help="美国常用信用评分，分数越高通常信用风险越低。",
            )
        with middle:
            dti = st.slider(
                "Debt-to-Income Ratio",
                min_value=0.0,
                max_value=60.0,
                value=18.0,
                step=0.5,
                help="负债收入比。数值越高，说明偿债压力越大。",
            )
            revol_util = st.slider(
                "Revolving Utilization",
                min_value=0.0,
                max_value=120.0,
                value=42.0,
                step=1.0,
                help="信用卡等循环授信额度使用率。高利用率通常代表资金压力。",
            )
            int_rate = st.slider(
                "Interest Rate",
                min_value=5.0,
                max_value=31.0,
                value=13.5,
                step=0.1,
                help="贷款利率。高利率常与更高风险定价相关。",
            )
        with right:
            term_months = st.selectbox(
                "Term",
                TERM_OPTIONS,
                index=0,
                format_func=lambda value: f"{value} months",
                help="贷款期限。长期限通常意味着更长风险暴露期。",
            )
            purpose = st.selectbox(
                "Purpose",
                PURPOSE_OPTIONS,
                index=0,
                help="贷款用途。模型会将不同用途作为类别特征处理。",
            )
            home_ownership = st.selectbox(
                "Home Ownership",
                HOME_OWNERSHIP_OPTIONS,
                index=0,
                help="住房状态，可反映客户资产稳定性。",
            )
            verification_status = st.selectbox(
                "Verification",
                VERIFICATION_OPTIONS,
                index=1,
                help="收入或信息验证状态，用于辅助判断申请可信度。",
            )
            lgd = st.slider(
                "Loss Given Default",
                min_value=0.20,
                max_value=0.90,
                value=BASE_LOSS_GIVEN_DEFAULT,
                step=0.05,
                help="违约损失率。预期损失 = 贷款金额 x 违约概率 x LGD。",
            )

        submitted = st.form_submit_button(
            "Run Assessment",
            help="点击后系统会运行模型预测、计算信用评分、审批建议和预期损失。",
        )

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
    with st.spinner("Running model prediction and decision rules..."):
        assessment = assess_applicant(profile, lgd=lgd)

    decision_badge(assessment.decision)
    assessment_metrics(assessment)
    probability_source_panel(assessment)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Risk Drivers")
        driver_list(assessment.drivers)
    with col2:
        st.markdown("### Decision Details")
        st.write(f"Loss given default: {assessment.loss_given_default:.0%}")
        st.write(f"Estimated exposure: ${profile.loan_amnt:,.0f}")
        st.write(f"Loan-to-income ratio: {profile.loan_amnt / max(profile.annual_inc, 1):.1%}")
