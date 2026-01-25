import streamlit as st


DECISION_CLASS = {
    "Approve": "risk-approve",
    "Manual Review": "risk-review",
    "Decline": "risk-decline",
}


def decision_badge(decision: str) -> None:
    css_class = DECISION_CLASS.get(decision, "risk-review")
    st.markdown(
        f'<span class="risk-pill {css_class}">{decision}</span>',
        unsafe_allow_html=True,
    )
