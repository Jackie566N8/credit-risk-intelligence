from config.credit_decision_config import BASE_LOSS_GIVEN_DEFAULT
from risk.risk_applicant_profile import ApplicantProfile, RiskAssessment
from risk.risk_credit_score import probability_to_credit_score, probability_to_risk_grade
from risk.risk_decision_rules import approval_decision
from risk.risk_driver_rules import risk_drivers
from risk.risk_model_predictor import predict_model_default_probability
from risk.risk_probability_calculator import estimate_default_probability


def assess_applicant(
    profile: ApplicantProfile,
    lgd: float = BASE_LOSS_GIVEN_DEFAULT,
) -> RiskAssessment:
    scorecard_probability = estimate_default_probability(profile)
    model_probability = predict_model_default_probability(profile)
    probability_default = model_probability if model_probability is not None else scorecard_probability
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
        scorecard_probability_default=scorecard_probability,
        model_probability_default=model_probability,
        probability_source="Random Forest model" if model_probability is not None else "Scorecard fallback",
    )
