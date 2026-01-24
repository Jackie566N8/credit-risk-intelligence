from io import StringIO
import re

import pandas as pd


def parse_bullet_section(lines: list[str], heading: str) -> dict[str, str]:
    start = find_heading_index(lines, heading)
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


def parse_table_after_heading(lines: list[str], heading: str) -> pd.DataFrame:
    start = find_heading_index(lines, heading)
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
    convert_numeric_columns(df)
    return df


def parse_feature_importance(lines: list[str]) -> pd.DataFrame:
    pattern = re.compile(r"^\d+\.\s+(.+):\s+([0-9.]+)$")
    rows = []
    for line in lines:
        match = pattern.match(line)
        if match:
            rows.append({"feature": match.group(1), "importance": float(match.group(2))})
    return pd.DataFrame(rows)


def convert_numeric_columns(df: pd.DataFrame) -> None:
    for column in df.columns:
        if column in {"metric", "confusion_matrix"}:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() == df[column].notna().sum():
            df[column] = converted


def find_heading_index(lines: list[str], heading: str) -> int | None:
    for index, line in enumerate(lines):
        if line.strip() == heading:
            return index
    return None
