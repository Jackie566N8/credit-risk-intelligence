from pathlib import Path

import pandas as pd
import streamlit as st


def show_result_status(comparison_df: pd.DataFrame, result_files: list[Path]) -> None:
    available_count = sum(path.exists() for path in result_files)
    st.caption(f"Loaded {available_count} model result files.")
    if comparison_df.empty:
        st.warning("No comparison result file found. Run `python modeling/train_credit_models.py` first.")
