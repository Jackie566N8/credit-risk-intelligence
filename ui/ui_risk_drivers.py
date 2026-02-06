import streamlit as st


def driver_list(drivers: list[str]) -> None:
    if not drivers:
        st.info("No major risk drivers detected.")
        return
    for driver in drivers:
        st.write(f"- {driver}")
