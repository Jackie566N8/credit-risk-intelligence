from credit_modeling_common import ModelSpec, parse_single_model_args, run_single_model
from sklearn.linear_model import LogisticRegression


MODEL_SPEC = ModelSpec(
    name="logistic_regression",
    display_name="Logistic Regression",
    estimator_factory=lambda random_state, n_jobs: LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    ),
)


def main() -> None:
    args = parse_single_model_args(
        description="Train the LendingClub logistic regression credit-risk model.",
        default_output_name=f"{MODEL_SPEC.name}_results.txt",
    )
    result = run_single_model(MODEL_SPEC, args)
    print(result.report_text)
    print(f"Saved results to: {result.output_path}")


if __name__ == "__main__":
    main()
