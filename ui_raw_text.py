from pathlib import Path

import streamlit as st


def show_raw_text(path: Path, text: str) -> None:
    with st.expander(f"Raw result file: {path.name}", expanded=False):
        st.code(text, language="text")
