from credit_modeling_common import ModelSpec, parse_single_model_args, run_single_model
from sklearn.ensemble import RandomForestClassifier


MODEL_SPEC = ModelSpec(
    name="random_forest",
    display_name="Random Forest",
    estimator_factory=lambda random_state, n_jobs: RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=30,
        n_jobs=n_jobs,
        random_state=random_state,
    ),
)


def main() -> None:
    args = parse_single_model_args(
        description="Train the LendingClub random forest credit-risk model.",
        default_output_name=f"{MODEL_SPEC.name}_results.txt",
    )
    result = run_single_model(MODEL_SPEC, args)
    print(result.report_text)
    print(f"Saved results to: {result.output_path}")


if __name__ == "__main__":
    main()
