from config.credit_decision_config import (
    APPROVAL_PD_THRESHOLD,
    APPROVAL_SCORE_THRESHOLD,
    REVIEW_PD_THRESHOLD,
    REVIEW_SCORE_THRESHOLD,
)


def approval_decision(probability_default: float, credit_score: int) -> str:
    if probability_default <= APPROVAL_PD_THRESHOLD and credit_score >= APPROVAL_SCORE_THRESHOLD:
        return "Approve"
    if probability_default <= REVIEW_PD_THRESHOLD and credit_score >= REVIEW_SCORE_THRESHOLD:
        return "Manual Review"
    return "Decline"
