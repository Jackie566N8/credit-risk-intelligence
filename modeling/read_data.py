import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

TARGET_COLUMN = "credit_risk"
PREVIEW_ROWS = 5

FEATURE_COLUMNS = [
    "status_checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "present_employment_since",
    "installment_rate_pct_income",
    "personal_status_and_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "people_liable_count",
    "telephone",
    "foreign_worker",
]

NUMERIC_COLUMNS = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct_income",
    "present_residence_since",
    "age_years",
    "existing_credits_count",
    "people_liable_count",
]

CATEGORICAL_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS
]

OPERATION_NAMES = {
    "load": "Load Data",
    "shape": "Show Shape",
    "columns": "List Columns",
    "types": "Split Columns",
    "preview": "Preview Rows",
    "target": "Target Count",
    "missing": "Missing Check",
}


def print_step(name: str) -> None:
    print(f"\n=== {name} ===")


def get_data_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])

    matches = sorted(Path.cwd().rglob("german_credit_raw.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "找不到 german_credit_raw.csv，请运行: "
        "python3 read_data.py <your_csv_path>"
    )


def load_data(path: Path) -> List[Dict[str, str]]:
    """读取德国信用风险数据集。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_columns(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def show_shape(rows: List[Dict[str, str]], columns: List[str]) -> None:
    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(columns)}")


def list_columns(columns: List[str]) -> None:
    for index, column in enumerate(columns, start=1):
        print(f"{index}. {column}")


def show_column_groups(columns: List[str]) -> None:
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in columns]

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    print("Numeric Columns:")
    for column in NUMERIC_COLUMNS:
        print(f"- {column}")

    print("\nCategorical Columns:")
    for column in CATEGORICAL_COLUMNS:
        print(f"- {column}")

    print(f"\nTarget Column: {TARGET_COLUMN}")


def preview_rows(rows: List[Dict[str, str]], limit: int = PREVIEW_ROWS) -> None:
    for row in rows[:limit]:
        print(row)


def show_target_count(rows: List[Dict[str, str]], target_column: str = TARGET_COLUMN) -> None:
    counts = Counter(row[target_column] for row in rows if target_column in row)
    for value, count in sorted(counts.items()):
        print(f"{target_column}={value}: {count}")


def show_missing_check(rows: List[Dict[str, str]], columns: List[str]) -> None:
    missing_counts = {
        column: sum(1 for row in rows if row.get(column, "") == "")
        for column in columns
    }
    missing_counts = {
        column: count
        for column, count in missing_counts.items()
        if count > 0
    }

    if not missing_counts:
        print("No missing values found.")
        return

    for column, count in missing_counts.items():
        print(f"{column}: {count}")


def main() -> None:
    print_step(OPERATION_NAMES["load"])
    data_path = get_data_path()
    rows = load_data(data_path)
    columns = get_columns(rows)
    print(f"Loaded from: {data_path}")

    print_step(OPERATION_NAMES["shape"])
    show_shape(rows, columns)

    print_step(OPERATION_NAMES["columns"])
    list_columns(columns)

    print_step(OPERATION_NAMES["types"])
    show_column_groups(columns)

    print_step(OPERATION_NAMES["preview"])
    preview_rows(rows)

    print_step(OPERATION_NAMES["target"])
    show_target_count(rows)

    print_step(OPERATION_NAMES["missing"])
    show_missing_check(rows, columns)


if __name__ == "__main__":
    main()
