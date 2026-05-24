import pandas as pd

from risk.risk_applicant_profile import ApplicantProfile


def build_batch_model_features(profiles: list[ApplicantProfile]) -> pd.DataFrame:
    if not profiles:
        return pd.DataFrame()
    return pd.concat([build_model_features(profile) for profile in profiles], ignore_index=True)


def build_model_features(profile: ApplicantProfile) -> pd.DataFrame:
    funded_amnt = profile.loan_amnt
    installment = monthly_installment(profile.loan_amnt, profile.int_rate, profile.term_months)
    grade, sub_grade = infer_grade(profile.fico_score, profile.int_rate)

    return pd.DataFrame(
        [
            {
                "loan_amnt": profile.loan_amnt,
                "funded_amnt": funded_amnt,
                "term_months": float(profile.term_months),
                "int_rate": profile.int_rate,
                "installment": installment,
                "annual_inc": profile.annual_inc,
                "dti": profile.dti,
                "fico_score": profile.fico_score,
                "open_acc": 11.0,
                "pub_rec": 0.0,
                "revol_bal": max(profile.loan_amnt * profile.revol_util / 100, 0.0),
                "revol_util": profile.revol_util,
                "total_acc": 25.0,
                "mort_acc": 1.0 if profile.home_ownership == "MORTGAGE" else 0.0,
                "pub_rec_bankruptcies": 0.0,
                "grade": grade,
                "sub_grade": sub_grade,
                "emp_length": "10+ years",
                "home_ownership": profile.home_ownership,
                "verification_status": profile.verification_status,
                "purpose": profile.purpose,
                "application_type": "Individual",
            }
        ]
    )


def monthly_installment(loan_amnt: float, int_rate: float, term_months: int) -> float:
    monthly_rate = int_rate / 100 / 12
    if monthly_rate <= 0:
        return loan_amnt / max(term_months, 1)
    return loan_amnt * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)


def infer_grade(fico_score: float, int_rate: float) -> tuple[str, str]:
    if fico_score >= 760 and int_rate <= 10:
        return "A", "A2"
    if fico_score >= 720 and int_rate <= 13:
        return "B", "B2"
    if fico_score >= 680 and int_rate <= 17:
        return "C", "C3"
    if fico_score >= 640 and int_rate <= 21:
        return "D", "D3"
    if fico_score >= 600:
        return "E", "E3"
    return "F", "F3"
