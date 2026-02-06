from io import StringIO
from pathlib import Path

import pandas as pd

from config.app_paths import COMPARISON_RESULTS_FILE, MODEL_RESULT_FILES
from config.model_display_config import MODEL_LABELS
from model_results.model_report_types import ModelReport
from model_results.model_result_parser import (
    parse_bullet_section,
    parse_feature_importance,
    parse_table_after_heading,
)


def load_comparison_results(path: Path = COMPARISON_RESULTS_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    table_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if "\t" in line]
    if not table_lines:
        return pd.DataFrame()

    df = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t")
    numeric_columns = [
        "cv_roc_auc_mean",
        "test_roc_auc",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "accuracy",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df


def load_all_model_reports() -> dict[str, ModelReport]:
    return {
        model_name: load_model_report(model_name, path)
        for model_name, path in MODEL_RESULT_FILES.items()
        if path.exists()
    }


def load_model_report(model_name: str, path: Path) -> ModelReport:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return ModelReport(
        name=model_name,
        label=MODEL_LABELS.get(model_name, model_name),
        path=path,
        raw_text=text,
        data_summary=parse_bullet_section(lines, "Data"),
        cv_metrics=parse_table_after_heading(lines, "Cross-Validation on Training Split"),
        holdout_metrics=parse_table_after_heading(lines, "Holdout Test Result with Optimized Threshold"),
        default_threshold_metrics=parse_table_after_heading(lines, "Default 0.50 Threshold Reference"),
        feature_importance=parse_feature_importance(lines),
    )
