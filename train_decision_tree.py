from credit_modeling_common import ModelSpec, parse_single_model_args, run_single_model
from sklearn.tree import DecisionTreeClassifier


MODEL_SPEC = ModelSpec(
    name="decision_tree",
    display_name="Decision Tree",
    estimator_factory=lambda random_state, n_jobs: DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=80,
        random_state=random_state,
    ),
)


def main() -> None:
    args = parse_single_model_args(
        description="Train the LendingClub decision tree credit-risk model.",
        default_output_name=f"{MODEL_SPEC.name}_results.txt",
    )
    result = run_single_model(MODEL_SPEC, args)
    print(result.report_text)
    print(f"Saved results to: {result.output_path}")


if __name__ == "__main__":
    main()
