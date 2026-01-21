from dataclasses import asdict, dataclass
from math import exp, log1p

import pandas as pd

from dashboard.config import (
    APPROVAL_PD_THRESHOLD,
    APPROVAL_SCORE_THRESHOLD,
    BASE_LOSS_GIVEN_DEFAULT,
    REVIEW_PD_THRESHOLD,
    REVIEW_SCORE_THRESHOLD,
)


@dataclass(frozen=True)
class ApplicantProfile:
    loan_amnt: float
    annual_inc: float
    fico_score: float
    dti: float
    revol_util: float
    int_rate: float
    term_months: int
    purpose: str
    home_ownership: str
    verification_status: str


@dataclass(frozen=True)
class RiskAssessment:
    probability_default: float
    credit_score: int
    risk_grade: str
    decision: str
    expected_loss: float
    loss_given_default: float
    drivers: list[str]


REQUIRED_BATCH_COLUMNS = list(ApplicantProfile.__dataclass_fields__.keys())


def assess_applicant(profile: ApplicantProfile, lgd: float = BASE_LOSS_GIVEN_DEFAULT) -> RiskAssessment:
    probability_default = _estimate_default_probability(profile)
    credit_score = _probability_to_score(probability_default)
    risk_grade = _risk_grade(probability_default)
    decision = _decision(probability_default, credit_score)
    expected_loss = profile.loan_amnt * probability_default * lgd
    drivers = _risk_drivers(profile, probability_default)
    return RiskAssessment(
        probability_default=probability_default,
        credit_score=credit_score,
        risk_grade=risk_grade,
        decision=decision,
        expected_loss=expected_loss,
        loss_given_default=lgd,
        drivers=drivers,
    )


def assess_batch(df: pd.DataFrame, lgd: float = BASE_LOSS_GIVEN_DEFAULT) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_BATCH_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    rows = []
    for _, row in df.iterrows():
        profile = ApplicantProfile(
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
        assessment = assess_applicant(profile, lgd=lgd)
        rows.append({**asdict(profile), **asdict(assessment), "drivers": "; ".join(assessment.drivers)})
    return pd.DataFrame(rows)


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


def _estimate_default_probability(profile: ApplicantProfile) -> float:
    score = -1.35
    score += (700 - profile.fico_score) / 85
    score += (profile.dti - 18) / 22
    score += (profile.revol_util - 45) / 65
    score += (profile.int_rate - 13) / 8
    score += 0.28 if profile.term_months >= 60 else -0.06
    score += _purpose_adjustment(profile.purpose)
    score += _home_adjustment(profile.home_ownership)
    score += _verification_adjustment(profile.verification_status)
    score += _loan_income_adjustment(profile.loan_amnt, profile.annual_inc)
    return min(max(_sigmoid(score), 0.02), 0.85)


def _probability_to_score(probability_default: float) -> int:
    score = round(850 - probability_default * 550)
    return int(min(max(score, 300), 850))


def _risk_grade(probability_default: float) -> str:
    if probability_default < 0.10:
        return "A"
    if probability_default < 0.18:
        return "B"
    if probability_default < 0.28:
        return "C"
    if probability_default < 0.40:
        return "D"
    return "E"


def _decision(probability_default: float, credit_score: int) -> str:
    if probability_default <= APPROVAL_PD_THRESHOLD and credit_score >= APPROVAL_SCORE_THRESHOLD:
        return "Approve"
    if probability_default <= REVIEW_PD_THRESHOLD and credit_score >= REVIEW_SCORE_THRESHOLD:
        return "Manual Review"
    return "Decline"


def _risk_drivers(profile: ApplicantProfile, probability_default: float) -> list[str]:
    drivers = []
    if profile.fico_score < 660:
        drivers.append("Low FICO score")
    if profile.dti > 30:
        drivers.append("High debt-to-income ratio")
    if profile.revol_util > 75:
        drivers.append("High revolving utilization")
    if profile.int_rate > 18:
        drivers.append("High interest rate")
    if profile.term_months >= 60:
        drivers.append("Long repayment term")
    if profile.loan_amnt / max(profile.annual_inc, 1) > 0.45:
        drivers.append("Large loan relative to income")
    if probability_default < 0.18 and not drivers:
        drivers.append("No major risk concentration")
    if probability_default >= 0.18 and not drivers:
        drivers.append("Borderline scorecard risk")
    return drivers[:5]


def _purpose_adjustment(purpose: str) -> float:
    return {
        "credit_card": -0.10,
        "debt_consolidation": 0.10,
        "home_improvement": -0.02,
        "major_purchase": -0.04,
        "medical": 0.12,
        "small_business": 0.24,
        "car": -0.08,
        "other": 0.08,
    }.get(purpose, 0.06)


def _home_adjustment(home_ownership: str) -> float:
    return {
        "MORTGAGE": -0.12,
        "OWN": -0.05,
        "RENT": 0.10,
        "OTHER": 0.08,
    }.get(home_ownership, 0.05)


def _verification_adjustment(verification_status: str) -> float:
    return {
        "Not Verified": -0.04,
        "Source Verified": 0.03,
        "Verified": 0.08,
    }.get(verification_status, 0.00)


def _loan_income_adjustment(loan_amnt: float, annual_inc: float) -> float:
    ratio = loan_amnt / max(annual_inc, 1)
    return min(max(log1p(ratio * 3) - 0.45, -0.20), 0.45)


def _sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))
