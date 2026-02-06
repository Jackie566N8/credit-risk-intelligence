from math import log1p


def purpose_adjustment(purpose: str) -> float:
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


def home_ownership_adjustment(home_ownership: str) -> float:
    return {
        "MORTGAGE": -0.12,
        "OWN": -0.05,
        "RENT": 0.10,
        "OTHER": 0.08,
    }.get(home_ownership, 0.05)


def verification_adjustment(verification_status: str) -> float:
    return {
        "Not Verified": -0.04,
        "Source Verified": 0.03,
        "Verified": 0.08,
    }.get(verification_status, 0.00)


def loan_income_adjustment(loan_amnt: float, annual_inc: float) -> float:
    ratio = loan_amnt / max(annual_inc, 1)
    return min(max(log1p(ratio * 3) - 0.45, -0.20), 0.45)
