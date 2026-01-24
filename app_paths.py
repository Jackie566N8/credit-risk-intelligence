from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELING_DIR = PROJECT_ROOT / "modeling"
RESULTS_DIR = MODELING_DIR / "results"
FIGURES_DIR = MODELING_DIR / "figures"

COMPARISON_RESULTS_FILE = RESULTS_DIR / "all_models_comparison_results.txt"

MODEL_RESULT_FILES = {
    "logistic_regression": RESULTS_DIR / "logistic_regression_results.txt",
    "decision_tree": RESULTS_DIR / "decision_tree_results.txt",
    "random_forest": RESULTS_DIR / "random_forest_results.txt",
    "gradient_boosting": RESULTS_DIR / "gradient_boosting_results.txt",
}
