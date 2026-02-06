from dataclasses import dataclass


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
