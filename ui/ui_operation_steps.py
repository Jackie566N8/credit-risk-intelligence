import streamlit as st


def operation_steps(title: str, steps: list[str]) -> None:
    with st.expander(f"{title} - 操作提示", expanded=True):
        for index, step in enumerate(steps, start=1):
            st.markdown(f"**Step {index}.** {step}")
