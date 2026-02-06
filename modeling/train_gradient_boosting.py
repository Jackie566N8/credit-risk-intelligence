from credit_modeling_common import ModelSpec, parse_single_model_args, run_single_model
from sklearn.ensemble import HistGradientBoostingClassifier


MODEL_SPEC = ModelSpec(
    name="gradient_boosting",
    display_name="Gradient Boosting",
    estimator_factory=lambda random_state, n_jobs: HistGradientBoostingClassifier(
        max_iter=140,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=random_state,
    ),
)


def main() -> None:
    args = parse_single_model_args(
        description="Train the LendingClub gradient boosting credit-risk model.",
        default_output_name=f"{MODEL_SPEC.name}_results.txt",
    )
    result = run_single_model(MODEL_SPEC, args)
    print(result.report_text)
    print(f"Saved results to: {result.output_path}")


if __name__ == "__main__":
    main()
