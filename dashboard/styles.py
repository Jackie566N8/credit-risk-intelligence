import streamlit as st


def apply_app_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
        }
        [data-testid="stMetricLabel"] {
            color: #4b5563;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }
        .risk-pill {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            font-weight: 700;
            border: 1px solid rgba(0, 0, 0, 0.08);
        }
        .risk-approve {
            background: #dcfce7;
            color: #166534;
        }
        .risk-review {
            background: #fef3c7;
            color: #92400e;
        }
        .risk-decline {
            background: #fee2e2;
            color: #991b1b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
