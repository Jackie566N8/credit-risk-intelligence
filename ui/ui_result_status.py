from pathlib import Path

import pandas as pd
import streamlit as st


def show_result_status(
    comparison_df: pd.DataFrame,
    result_files: list[Path],
    artifact_files: list[Path] | None = None,
) -> None:
    available_count = sum(path.exists() for path in result_files)
    artifact_count = sum(path.exists() for path in artifact_files or [])
    if comparison_df.empty:
        st.warning("No comparison result file found. Run `python modeling/train_credit_models.py` first.")
        return

    if available_count == len(result_files) and artifact_count > 0:
        st.success(
            f"Loaded {available_count} model result files and {artifact_count} model artifact."
        )
    else:
        st.info(
            f"Loaded {available_count}/{len(result_files)} model result files and "
            f"{artifact_count}/{len(artifact_files or [])} model artifacts."
        )
