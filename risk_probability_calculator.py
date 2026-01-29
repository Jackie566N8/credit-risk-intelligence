from math import exp

from risk_applicant_profile import ApplicantProfile
from risk_probability_adjustments import (
    home_ownership_adjustment,
    loan_income_adjustment,
    purpose_adjustment,
    verification_adjustment,
)


def estimate_default_probability(profile: ApplicantProfile) -> float:
    score = -1.35
    score += (700 - profile.fico_score) / 85
    score += (profile.dti - 18) / 22
    score += (profile.revol_util - 45) / 65
    score += (profile.int_rate - 13) / 8
    score += 0.28 if profile.term_months >= 60 else -0.06
    score += purpose_adjustment(profile.purpose)
    score += home_ownership_adjustment(profile.home_ownership)
    score += verification_adjustment(profile.verification_status)
    score += loan_income_adjustment(profile.loan_amnt, profile.annual_inc)
    return min(max(_sigmoid(score), 0.02), 0.85)


def _sigmoid(value: float) -> float:
    return 1 / (1 + exp(-value))
