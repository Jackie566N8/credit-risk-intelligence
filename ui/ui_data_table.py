import pandas as pd
import streamlit as st


def numeric_table(df: pd.DataFrame) -> None:
    if df.empty:
        return

    display_df = df.copy()
    numeric_columns = display_df.select_dtypes(include=["number"]).columns
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(column, format="%.4f")
            for column in numeric_columns
        },
    )
