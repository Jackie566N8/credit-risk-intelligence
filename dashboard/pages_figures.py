from pathlib import Path

import streamlit as st

from dashboard.components import page_header
from dashboard.config import FIGURE_GROUPS, FIGURES_DIR


def render_figures_page() -> None:
    page_header(
        "EDA Figures",
        "Seaborn and Matplotlib charts generated from accepted and rejected LendingClub records.",
    )
    if not FIGURES_DIR.exists():
        st.warning("No figures directory found. Run `python modeling/plot.py` first.")
        return

    for group_name, file_names in FIGURE_GROUPS.items():
        st.markdown(f"### {group_name}")
        existing_paths = [FIGURES_DIR / file_name for file_name in file_names if (FIGURES_DIR / file_name).exists()]
        if not existing_paths:
            st.info(f"No figures available for {group_name}.")
            continue

        _image_grid(existing_paths)


def _image_grid(paths: list[Path]) -> None:
    for index in range(0, len(paths), 2):
        cols = st.columns(2)
        for col, path in zip(cols, paths[index : index + 2]):
            with col:
                st.image(str(path), caption=path.name, use_container_width=True)
