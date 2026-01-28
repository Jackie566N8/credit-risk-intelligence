from dataclasses import asdict

import pandas as pd

from credit_decision_config import BASE_LOSS_GIVEN_DEFAULT
from risk_applicant_profile import ApplicantProfile, REQUIRED_BATCH_COLUMNS
from risk_single_assessment import assess_applicant


def assess_batch(df: pd.DataFrame, lgd: float = BASE_LOSS_GIVEN_DEFAULT) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_BATCH_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    rows = []
    for _, row in df.iterrows():
        profile = applicant_from_row(row)
        assessment = assess_applicant(profile, lgd=lgd)
        rows.append({**asdict(profile), **asdict(assessment), "drivers": "; ".join(assessment.drivers)})
    return pd.DataFrame(rows)


def applicant_from_row(row: pd.Series) -> ApplicantProfile:
    return ApplicantProfile(
        loan_amnt=float(row["loan_amnt"]),
        annual_inc=float(row["annual_inc"]),
        fico_score=float(row["fico_score"]),
        dti=float(row["dti"]),
        revol_util=float(row["revol_util"]),
        int_rate=float(row["int_rate"]),
        term_months=int(row["term_months"]),
        purpose=str(row["purpose"]),
        home_ownership=str(row["home_ownership"]),
        verification_status=str(row["verification_status"]),
    )
