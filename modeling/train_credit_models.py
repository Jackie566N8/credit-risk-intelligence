from credit_modeling_common import parse_all_models_args, run_many_models
from train_decision_tree import MODEL_SPEC as DECISION_TREE_SPEC
from train_gradient_boosting import MODEL_SPEC as GRADIENT_BOOSTING_SPEC
from train_logistic_regression import MODEL_SPEC as LOGISTIC_REGRESSION_SPEC
from train_random_forest import MODEL_SPEC as RANDOM_FOREST_SPEC


MODEL_SPECS = [
    LOGISTIC_REGRESSION_SPEC,
    DECISION_TREE_SPEC,
    RANDOM_FOREST_SPEC,
    GRADIENT_BOOSTING_SPEC,
]


def main() -> None:
    args = parse_all_models_args(
        description="Train all LendingClub credit-risk models and save separate result files."
    )
    results = run_many_models(MODEL_SPECS, args)
    for result in results:
        print(f"Saved {result.display_name} results to: {result.output_path}")
    print(f"Saved comparison results to: {args.results_dir / 'all_models_comparison_results.txt'}")


if __name__ == "__main__":
    main()
