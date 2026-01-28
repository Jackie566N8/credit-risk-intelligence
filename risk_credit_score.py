def probability_to_credit_score(probability_default: float) -> int:
    score = round(850 - probability_default * 550)
    return int(min(max(score, 300), 850))


def probability_to_risk_grade(probability_default: float) -> str:
    if probability_default < 0.10:
        return "A"
    if probability_default < 0.18:
        return "B"
    if probability_default < 0.28:
        return "C"
    if probability_default < 0.40:
        return "D"
    return "E"
