from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re

import pandas as pd

from dashboard.config import COMPARISON_RESULTS_FILE, MODEL_LABELS, MODEL_RESULT_FILES


@dataclass
class ModelReport:
    name: str
    label: str
    path: Path
    raw_text: str
    data_summary: dict[str, str]
    cv_metrics: pd.DataFrame
    holdout_metrics: pd.DataFrame
    default_threshold_metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def load_comparison_results(path: Path = COMPARISON_RESULTS_FILE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = []
    for line in lines:
        if "\t" in line:
            table_lines.append(line)

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
        data_summary=_parse_bullet_section(lines, "Data"),
        cv_metrics=_parse_table_after_heading(lines, "Cross-Validation on Training Split"),
        holdout_metrics=_parse_table_after_heading(lines, "Holdout Test Result with Optimized Threshold"),
        default_threshold_metrics=_parse_table_after_heading(lines, "Default 0.50 Threshold Reference"),
        feature_importance=_parse_feature_importance(lines),
    )


def _parse_bullet_section(lines: list[str], heading: str) -> dict[str, str]:
    start = _find_heading_index(lines, heading)
    if start is None:
        return {}

    values: dict[str, str] = {}
    for line in lines[start + 1 :]:
        if not line:
            break
        if not line.startswith("- "):
            continue
        key, _, value = line[2:].partition(": ")
        if key and value:
            values[key] = value
    return values


def _parse_table_after_heading(lines: list[str], heading: str) -> pd.DataFrame:
    start = _find_heading_index(lines, heading)
    if start is None:
        return pd.DataFrame()

    table_lines = []
    for line in lines[start + 1 :]:
        if not line:
            if table_lines:
                break
            continue
        if "\t" in line:
            table_lines.append(line)

    if not table_lines:
        return pd.DataFrame()

    df = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t")
    for column in df.columns:
        if column in {"metric", "confusion_matrix"}:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() == df[column].notna().sum():
            df[column] = converted
    return df


def _parse_feature_importance(lines: list[str]) -> pd.DataFrame:
    pattern = re.compile(r"^\d+\.\s+(.+):\s+([0-9.]+)$")
    rows = []
    for line in lines:
        match = pattern.match(line)
        if match:
            rows.append(
                {
                    "feature": match.group(1),
                    "importance": float(match.group(2)),
                }
            )
    return pd.DataFrame(rows)


def _find_heading_index(lines: list[str], heading: str) -> int | None:
    for index, line in enumerate(lines):
        if line.strip() == heading:
            return index
    return None
