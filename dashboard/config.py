from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

COMPARISON_RESULTS_FILE = RESULTS_DIR / "all_models_comparison_results.txt"

MODEL_RESULT_FILES = {
    "logistic_regression": RESULTS_DIR / "logistic_regression_results.txt",
    "decision_tree": RESULTS_DIR / "decision_tree_results.txt",
    "random_forest": RESULTS_DIR / "random_forest_results.txt",
    "gradient_boosting": RESULTS_DIR / "gradient_boosting_results.txt",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

PAGE_LABELS = {
    "Portfolio Overview": "Portfolio Overview",
    "Model Detail": "Model Detail",
    "Single Customer Decision": "Single Customer Decision",
    "Batch Risk Assessment": "Batch Risk Assessment",
    "EDA Figures": "EDA Figures",
}

METRIC_LABELS = {
    "cv_roc_auc_mean": "CV ROC-AUC",
    "test_roc_auc": "Test ROC-AUC",
    "average_precision": "Average Precision",
    "threshold": "Threshold",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "f2": "F2",
    "accuracy": "Accuracy",
}

PURPOSE_OPTIONS = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "medical",
    "car",
    "other",
]

HOME_OWNERSHIP_OPTIONS = ["MORTGAGE", "RENT", "OWN", "OTHER"]
VERIFICATION_OPTIONS = ["Not Verified", "Source Verified", "Verified"]
TERM_OPTIONS = [36, 60]

BASE_LOSS_GIVEN_DEFAULT = 0.55
APPROVAL_PD_THRESHOLD = 0.18
REVIEW_PD_THRESHOLD = 0.35
APPROVAL_SCORE_THRESHOLD = 660
REVIEW_SCORE_THRESHOLD = 580

FIGURE_GROUPS = {
    "Accepted Loans": [
        "target_count.png",
        "grade_default_rate.png",
        "fico_band_default_rate.png",
        "term_grade_default_rate.png",
        "income_loan_amount_default_heatmap.png",
        "purpose_default_rate.png",
        "interest_rate_by_grade.png",
        "revol_util_by_target.png",
        "fico_dti_by_target.png",
        "numeric_correlation_heatmap.png",
        "numeric_distributions.png",
        "loan_amount_by_target.png",
    ],
    "Rejected Applications": [
        "rejected_risk_score_distribution.png",
        "rejected_amount_vs_dti.png",
        "top_rejected_states.png",
        "rejected_applications_by_year.png",
    ],
}
