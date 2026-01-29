from risk_applicant_profile import ApplicantProfile


def risk_drivers(profile: ApplicantProfile, probability_default: float) -> list[str]:
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
