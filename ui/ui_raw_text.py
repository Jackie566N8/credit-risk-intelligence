from pathlib import Path

import streamlit as st


def show_raw_text(path: Path, text: str) -> None:
    display_text = sanitize_local_paths(text)
    with st.expander(f"Raw result file: {path.name}", expanded=False):
        st.code(display_text, language="text")


def sanitize_local_paths(text: str) -> str:
    project_root = Path(__file__).resolve().parents[1].as_posix()
    cwd = Path.cwd().as_posix()
    sanitized = text.replace(project_root, "<project-root>")
    if cwd != project_root:
        sanitized = sanitized.replace(cwd, "<project-root>")
    return sanitized
