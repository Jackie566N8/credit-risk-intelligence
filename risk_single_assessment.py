from credit_decision_config import BASE_LOSS_GIVEN_DEFAULT
from risk_applicant_profile import ApplicantProfile, RiskAssessment
from risk_credit_score import probability_to_credit_score, probability_to_risk_grade
from risk_decision_rules import approval_decision
from risk_driver_rules import risk_drivers
from risk_probability_calculator import estimate_default_probability


def assess_applicant(
    profile: ApplicantProfile,
    lgd: float = BASE_LOSS_GIVEN_DEFAULT,
) -> RiskAssessment:
    probability_default = estimate_default_probability(profile)
    credit_score = probability_to_credit_score(probability_default)
    risk_grade = probability_to_risk_grade(probability_default)
    decision = approval_decision(probability_default, credit_score)
    expected_loss = profile.loan_amnt * probability_default * lgd
    drivers = risk_drivers(profile, probability_default)
    return RiskAssessment(
        probability_default=probability_default,
        credit_score=credit_score,
        risk_grade=risk_grade,
        decision=decision,
        expected_loss=expected_loss,
        loss_given_default=lgd,
        drivers=drivers,
    )
