from dataclasses import asdict

import pandas as pd

from config.credit_decision_config import BASE_LOSS_GIVEN_DEFAULT
from risk.risk_applicant_profile import ApplicantProfile, REQUIRED_BATCH_COLUMNS, RiskAssessment
from risk.risk_credit_score import probability_to_credit_score, probability_to_risk_grade
from risk.risk_decision_rules import approval_decision
from risk.risk_driver_rules import risk_drivers
from risk.risk_model_predictor import predict_model_default_probabilities
from risk.risk_probability_calculator import estimate_default_probability


def assess_batch(df: pd.DataFrame, lgd: float = BASE_LOSS_GIVEN_DEFAULT) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_BATCH_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    profiles = [applicant_from_row(row) for _, row in df.iterrows()]
    scorecard_probabilities = [estimate_default_probability(profile) for profile in profiles]
    model_probabilities = predict_model_default_probabilities(profiles)

    rows = []
    for index, profile in enumerate(profiles):
        assessment = assessment_from_probabilities(
            profile=profile,
            scorecard_probability=scorecard_probabilities[index],
            model_probability=None if model_probabilities is None else model_probabilities[index],
            lgd=lgd,
        )
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


def assessment_from_probabilities(
    profile: ApplicantProfile,
    scorecard_probability: float,
    model_probability: float | None,
    lgd: float,
) -> RiskAssessment:
    probability_default = model_probability if model_probability is not None else scorecard_probability
    credit_score = probability_to_credit_score(probability_default)
    risk_grade = probability_to_risk_grade(probability_default)
    decision = approval_decision(probability_default, credit_score)
    expected_loss = profile.loan_amnt * probability_default * lgd
    return RiskAssessment(
        probability_default=probability_default,
        credit_score=credit_score,
        risk_grade=risk_grade,
        decision=decision,
        expected_loss=expected_loss,
        loss_given_default=lgd,
        drivers=risk_drivers(profile, probability_default),
        scorecard_probability_default=scorecard_probability,
        model_probability_default=model_probability,
        probability_source="Random Forest model" if model_probability is not None else "Scorecard fallback",
    )
