from pathlib import Path

import streamlit as st

from app_paths import FIGURES_DIR
from figure_gallery_config import FIGURE_GROUPS
from ui_page_header import page_header


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
        paths = [FIGURES_DIR / file_name for file_name in file_names if (FIGURES_DIR / file_name).exists()]
        if not paths:
            st.info(f"No figures available for {group_name}.")
            continue
        image_grid(paths)


def image_grid(paths: list[Path]) -> None:
    for index in range(0, len(paths), 2):
        cols = st.columns(2)
        for col, path in zip(cols, paths[index : index + 2]):
            with col:
                st.image(str(path), caption=path.name, use_container_width=True)
