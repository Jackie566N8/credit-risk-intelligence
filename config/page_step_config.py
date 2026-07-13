PAGE_STEPS = {
    "Portfolio Overview": [
        "Confirm that the top status shows Ready for demo, which means result files and model artifacts are loaded.",
        "Review the Project Report Highlight to explain the report-level random forest metrics.",
        "Review Reproducible Training Results to compare the current runnable experiment metrics.",
        "Switch the Metric dropdown to compare models by ROC-AUC, recall, F2, and other metrics.",
    ],
    "Model Detail": [
        "Select a model from the Model dropdown.",
        "Review Cross Validation to evaluate model stability.",
        "Review Holdout Test to compare the optimized threshold with the default 0.50 threshold.",
        "Review Feature Importance to explain the main risk factors.",
    ],
    "Single Customer Decision": [
        "Enter loan amount, income, FICO, DTI, revolving utilization, interest rate, and loan term.",
        "Confirm loan purpose, home ownership, verification status, and the LGD parameter.",
        "Click Run Assessment and wait for model prediction and decision rules.",
        "Review the approval recommendation, credit score, expected loss, and main risk drivers.",
    ],
    "Batch Risk Assessment": [
        "Upload a CSV with the required fields; if no file is uploaded, sample applications are used.",
        "Check Required columns and make sure the uploaded field names match.",
        "Wait for batch scoring, then review the decision distribution and PD-score scatter plot.",
        "Download the scored result CSV or the CSV template.",
    ],
    "EDA Figures": [
        "Confirm that `python modeling/plot.py` has been run to generate the figures.",
        "Start with Accepted Loans figures to explain default labels, FICO, DTI, and credit grades.",
        "Then review Rejected Applications figures to describe applicant-side risk profiles.",
        "Connect the chart findings with model feature importance to explain the business logic.",
    ],
}
