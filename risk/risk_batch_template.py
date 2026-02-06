import pandas as pd


def sample_batch_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "loan_amnt": 12000,
                "annual_inc": 78000,
                "fico_score": 720,
                "dti": 18.5,
                "revol_util": 38.0,
                "int_rate": 11.5,
                "term_months": 36,
                "purpose": "debt_consolidation",
                "home_ownership": "MORTGAGE",
                "verification_status": "Source Verified",
            },
            {
                "loan_amnt": 28000,
                "annual_inc": 52000,
                "fico_score": 635,
                "dti": 31.0,
                "revol_util": 82.0,
                "int_rate": 21.9,
                "term_months": 60,
                "purpose": "small_business",
                "home_ownership": "RENT",
                "verification_status": "Verified",
            },
        ]
    )
