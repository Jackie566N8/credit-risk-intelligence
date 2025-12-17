import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
import warnings

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    logical_cpus = os.cpu_count() or 1
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, logical_cpus - 1))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/accepted/accepted_2007_to_2018Q4.csv")
RESULTS_DIR = Path("results")
TARGET_COLUMN = "default_flag"
STATUS_COLUMN = "loan_status"

DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",
}

NON_DEFAULT_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

RAW_COLUMNS = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    STATUS_COLUMN,
    "purpose",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "application_type",
]

NUMERIC_FEATURES = [
    "loan_amnt",
    "funded_amnt",
    "term_months",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_score",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
]

CATEGORICAL_FEATURES = [
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

READ_DTYPE = {
    "loan_amnt": "float32",
    "funded_amnt": "float32",
    "installment": "float32",
    "annual_inc": "float32",
    "dti": "float32",
    "fico_range_low": "float32",
    "fico_range_high": "float32",
    "open_acc": "float32",
    "pub_rec": "float32",
    "revol_bal": "float32",
    "total_acc": "float32",
    "mort_acc": "float32",
    "pub_rec_bankruptcies": "float32",
    "term": "category",
    "grade": "category",
    "sub_grade": "category",
    "emp_length": "category",
    "home_ownership": "category",
    "verification_status": "category",
    STATUS_COLUMN: "category",
    "purpose": "category",
    "application_type": "category",
}

SCORING = {
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    estimator_factory: Callable[[int, int], object]


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    accuracy: float


@dataclass
class DatasetBundle:
    raw_rows: int
    excluded_rows: int
    df: pd.DataFrame
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    x_fit: pd.DataFrame
    x_threshold: pd.DataFrame
    y_fit: pd.Series
    y_threshold: pd.Series


@dataclass
class ModelTrainingResult:
    model_name: str
    display_name: str
    output_path: Path
    cv_summary: dict[str, tuple[float, float]]
    roc_auc: float
    average_precision: float
    default_threshold: ThresholdResult
    optimized_threshold: ThresholdResult
    confusion_matrix: str
    report_text: str


def clean_percent_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float32")

    cleaned = series.astype("string").str.strip().str.rstrip("%")
    return pd.to_numeric(cleaned, errors="coerce").astype("float32")


def load_model_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path, usecols=RAW_COLUMNS, dtype=READ_DTYPE)
    raw_rows = len(df)

    df["int_rate"] = clean_percent_column(df["int_rate"])
    df["revol_util"] = clean_percent_column(df["revol_util"])
    df["term_months"] = (
        df["term"].astype("string").str.extract(r"(\d+)", expand=False).astype("float32")
    )
    df["fico_score"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).astype("float32")

    df[TARGET_COLUMN] = np.nan
    default_mask = df[STATUS_COLUMN].isin(DEFAULT_STATUSES)
    non_default_mask = df[STATUS_COLUMN].isin(NON_DEFAULT_STATUSES)
    df.loc[default_mask, TARGET_COLUMN] = 1
    df.loc[non_default_mask, TARGET_COLUMN] = 0

    df = df[df[TARGET_COLUMN].notna()].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")
    df.attrs["raw_rows"] = raw_rows
    df.attrs["excluded_rows"] = raw_rows - len(df)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def stratified_sample(df: pd.DataFrame, sample_rows: int, random_state: int) -> pd.DataFrame:
    if sample_rows <= 0 or len(df) <= sample_rows:
        return df

    sampled, _ = train_test_split(
        df,
        train_size=sample_rows,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )
    return sampled.sort_index()


def prepare_dataset(args: argparse.Namespace) -> DatasetBundle:
    df = load_model_data(args.data_path)
    raw_rows = df.attrs.get("raw_rows", len(df))
    excluded_rows = df.attrs.get("excluded_rows", 0)
    df = stratified_sample(df, args.sample_rows, args.random_state)

    y = df[TARGET_COLUMN]
    x = df[FEATURE_COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    x_fit, x_threshold, y_fit, y_threshold = train_test_split(
        x_train,
        y_train,
        test_size=args.threshold_size,
        stratify=y_train,
        random_state=args.random_state,
    )
    return DatasetBundle(
        raw_rows=raw_rows,
        excluded_rows=excluded_rows,
        df=df,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        x_fit=x_fit,
        x_threshold=x_threshold,
        y_fit=y_fit,
        y_threshold=y_threshold,
    )


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(model: object, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
            ("model", model),
        ]
    )


def positive_probability(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(x)[:, 1]

    scores = pipeline.decision_function(x)
    return 1 / (1 + np.exp(-scores))


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> ThresholdResult:
    y_pred = (probabilities >= threshold).astype(int)
    return ThresholdResult(
        threshold=threshold,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        f2=fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
    )


def optimize_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> ThresholdResult:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.951, 0.01)

    results = [metrics_at_threshold(y_true, probabilities, float(t)) for t in thresholds]
    return max(results, key=lambda item: (item.f2, item.recall, item.f1))


def summarize_cv(cv_scores: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    summary = {}
    for metric in SCORING:
        values = cv_scores[f"test_{metric}"]
        summary[metric] = (float(np.mean(values)), float(np.std(values)))
    return summary


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def confusion_matrix_text(y_true: pd.Series, y_pred: np.ndarray) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    preprocess = pipeline.named_steps["preprocess"]
    names = preprocess.get_feature_names_out()
    return np.array([name.replace("numeric__", "").replace("categorical__", "") for name in names])


def feature_importance_lines(
    model_spec: ModelSpec,
    pipeline: Pipeline,
    top_n: int = 20,
) -> list[str]:
    model = pipeline.named_steps["model"]
    names = get_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"{model_spec.display_name} feature_importances_"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"{model_spec.display_name} absolute coefficients"
    else:
        return [f"{model_spec.display_name}: feature importance is not exposed by this estimator."]

    order = np.argsort(importances)[::-1][:top_n]
    lines = [title]
    for rank, index in enumerate(order, start=1):
        lines.append(f"{rank:02d}. {names[index]}: {importances[index]:.6f}")
    return lines


def format_single_model_report(
    model_spec: ModelSpec,
    args: argparse.Namespace,
    bundle: DatasetBundle,
    cv_summary: dict[str, tuple[float, float]],
    roc_auc: float,
    average_precision: float,
    default_result: ThresholdResult,
    optimized_result: ThresholdResult,
    confusion_text: str,
    feature_lines: list[str],
) -> str:
    y = bundle.df[TARGET_COLUMN]
    lines: list[str] = []
    lines.append(f"Credit Risk Modeling Results - {model_spec.display_name}")
    lines.append(f"Model name: {model_spec.name}")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Data")
    lines.append(f"- Source: {args.data_path}")
    lines.append(f"- Raw accepted rows: {bundle.raw_rows}")
    lines.append(f"- Excluded non-final/current-status rows: {bundle.excluded_rows}")
    lines.append(f"- Modeling rows used: {len(bundle.df)}")
    lines.append(f"- Default count: {int(y.sum())}")
    lines.append(f"- Non-default count: {int((y == 0).sum())}")
    lines.append(f"- Default rate: {y.mean():.4%}")
    lines.append(f"- Numeric features: {', '.join(NUMERIC_FEATURES)}")
    lines.append(f"- Categorical features: {', '.join(CATEGORICAL_FEATURES)}")
    lines.append("")
    lines.append("Training Setup")
    lines.append(f"- Model: {model_spec.display_name}")
    lines.append("- Class imbalance handling: SMOTE inside imbalanced-learn Pipeline")
    lines.append(f"- Cross validation: Stratified {args.cv_folds}-fold on training split")
    lines.append(f"- Test size: {args.test_size:.2f}")
    lines.append(f"- Threshold validation size inside training split: {args.threshold_size:.2f}")
    lines.append("- Threshold objective: maximize F2 score on validation split")
    lines.append("")
    lines.append("Cross-Validation on Training Split")
    lines.append("metric\tmean\tstd")
    for metric in ["roc_auc", "recall", "precision", "f1", "accuracy"]:
        mean, std = cv_summary[metric]
        lines.append(f"{metric}\t{format_metric(mean)}\t{format_metric(std)}")
    lines.append("")
    lines.append("Holdout Test Result with Optimized Threshold")
    lines.append("roc_auc\taverage_precision\tthreshold\tprecision\trecall\tf1\tf2\taccuracy\tconfusion_matrix")
    lines.append(
        "\t".join(
            [
                format_metric(roc_auc),
                format_metric(average_precision),
                format_metric(optimized_result.threshold),
                format_metric(optimized_result.precision),
                format_metric(optimized_result.recall),
                format_metric(optimized_result.f1),
                format_metric(optimized_result.f2),
                format_metric(optimized_result.accuracy),
                confusion_text,
            ]
        )
    )
    lines.append("")
    lines.append("Default 0.50 Threshold Reference")
    lines.append("precision\trecall\tf1\tf2\taccuracy")
    lines.append(
        "\t".join(
            [
                format_metric(default_result.precision),
                format_metric(default_result.recall),
                format_metric(default_result.f1),
                format_metric(default_result.f2),
                format_metric(default_result.accuracy),
            ]
        )
    )
    lines.append("")
    lines.extend(feature_lines)
    return "\n".join(lines) + "\n"


def train_model(
    model_spec: ModelSpec,
    args: argparse.Namespace,
    bundle: DatasetBundle,
    output_path: Path,
) -> ModelTrainingResult:
    warnings.filterwarnings("ignore", category=FutureWarning)
    pipeline = make_pipeline(
        model_spec.estimator_factory(args.random_state, args.n_jobs),
        args.random_state,
    )
    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    cv_scores = cross_validate(
        pipeline,
        bundle.x_train,
        bundle.y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
    )
    cv_summary = summarize_cv(cv_scores)

    threshold_pipeline = clone(pipeline)
    threshold_pipeline.fit(bundle.x_fit, bundle.y_fit)
    threshold_prob = positive_probability(threshold_pipeline, bundle.x_threshold)
    selected_threshold = optimize_threshold(bundle.y_threshold, threshold_prob).threshold

    final_pipeline = clone(pipeline)
    final_pipeline.fit(bundle.x_train, bundle.y_train)
    test_prob = positive_probability(final_pipeline, bundle.x_test)
    default_result = metrics_at_threshold(bundle.y_test, test_prob, 0.5)
    optimized_result = metrics_at_threshold(bundle.y_test, test_prob, selected_threshold)
    optimized_pred = (test_prob >= selected_threshold).astype(int)
    confusion_text = confusion_matrix_text(bundle.y_test, optimized_pred)
    roc_auc = roc_auc_score(bundle.y_test, test_prob)
    average_precision = average_precision_score(bundle.y_test, test_prob)
    feature_lines = feature_importance_lines(model_spec, final_pipeline)

    report_text = format_single_model_report(
        model_spec=model_spec,
        args=args,
        bundle=bundle,
        cv_summary=cv_summary,
        roc_auc=roc_auc,
        average_precision=average_precision,
        default_result=default_result,
        optimized_result=optimized_result,
        confusion_text=confusion_text,
        feature_lines=feature_lines,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return ModelTrainingResult(
        model_name=model_spec.name,
        display_name=model_spec.display_name,
        output_path=output_path,
        cv_summary=cv_summary,
        roc_auc=roc_auc,
        average_precision=average_precision,
        default_threshold=default_result,
        optimized_threshold=optimized_result,
        confusion_matrix=confusion_text,
        report_text=report_text,
    )


def run_single_model(model_spec: ModelSpec, args: argparse.Namespace) -> ModelTrainingResult:
    bundle = prepare_dataset(args)
    return train_model(model_spec, args, bundle, args.output_path)


def run_many_models(
    model_specs: list[ModelSpec],
    args: argparse.Namespace,
) -> list[ModelTrainingResult]:
    bundle = prepare_dataset(args)
    results = []
    for model_spec in model_specs:
        output_path = args.results_dir / f"{model_spec.name}_results.txt"
        results.append(train_model(model_spec, args, bundle, output_path))
    write_comparison_report(results, args.results_dir / "all_models_comparison_results.txt")
    return results


def write_comparison_report(results: list[ModelTrainingResult], output_path: Path) -> None:
    lines = ["Credit Risk Model Comparison", f"Generated at: {datetime.now().isoformat(timespec='seconds')}", ""]
    lines.append("model\tcv_roc_auc_mean\ttest_roc_auc\tthreshold\tprecision\trecall\tf1\tf2\taccuracy\tresult_file")
    for result in results:
        optimized = result.optimized_threshold
        lines.append(
            "\t".join(
                [
                    result.model_name,
                    format_metric(result.cv_summary["roc_auc"][0]),
                    format_metric(result.roc_auc),
                    format_metric(optimized.threshold),
                    format_metric(optimized.precision),
                    format_metric(optimized.recall),
                    format_metric(optimized.f1),
                    format_metric(optimized.f2),
                    format_metric(optimized.accuracy),
                    str(result.output_path),
                ]
            )
        )
    if results:
        best_auc = max(results, key=lambda item: item.cv_summary["roc_auc"][0])
        best_recall = max(results, key=lambda item: item.optimized_threshold.recall)
        lines.append("")
        lines.append(f"Best CV ROC-AUC model: {best_auc.model_name}")
        lines.append(f"Highest optimized holdout recall model: {best_recall.model_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=120_000,
        help="Stratified modeling sample size. Use 0 to train on all final-status rows.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def parse_single_model_args(description: str, default_output_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser)
    parser.add_argument("--output-path", type=Path, default=RESULTS_DIR / default_output_name)
    return parser.parse_args()


def parse_all_models_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()
