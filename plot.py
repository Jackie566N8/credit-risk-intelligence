from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/accepted/accepted_2007_to_2018Q4.csv")
FIGURES_DIR = Path("figures")
TARGET_COLUMN = "credit_risk"

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
    "status_checking_account",
    "credit_history",
    "purpose",
    "savings_account_bonds",
    "present_employment_since",
    "personal_status_and_sex",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]


def print_step(name: str) -> None:
    print(f"\n=== {name} ===")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path)
    df["target_label"] = df[TARGET_COLUMN].map({1: "Good", 2: "Bad"})
    df["default_flag"] = (df[TARGET_COLUMN] == 2).astype(int)
    return df


def show_basic_eda(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())
    print("\nNumeric summary:")
    print(df[NUMERIC_COLUMNS].describe().T)
    print("\nTarget count:")
    print(df["target_label"].value_counts())


def save_target_count_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "target_count.png"

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target_label", order=["Good", "Bad"])
    plt.title("Credit Risk Target Count")
    plt.xlabel("Credit Risk")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_numeric_histograms(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "numeric_distributions.png"

    df[NUMERIC_COLUMNS].hist(figsize=(12, 8), bins=25)
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_credit_amount_boxplot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "credit_amount_by_target.png"

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="target_label", y="credit_amount", order=["Good", "Bad"])
    plt.title("Credit Amount by Credit Risk")
    plt.xlabel("Credit Risk")
    plt.ylabel("Credit Amount")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_correlation_heatmap(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "numeric_correlation_heatmap.png"

    corr = df[NUMERIC_COLUMNS + ["default_flag"]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Numeric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_purpose_default_rate_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "purpose_default_rate.png"

    purpose_default = (
        df.groupby("purpose", as_index=False)["default_flag"]
        .mean()
        .sort_values("default_flag", ascending=False)
    )

    plt.figure(figsize=(9, 4))
    sns.barplot(data=purpose_default, x="purpose", y="default_flag")
    plt.title("Default Rate by Purpose")
    plt.xlabel("Purpose")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main() -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print_step("Load Data")
    df = load_data()
    print(f"Loaded from: {DATA_PATH}")

    print_step("Basic EDA")
    show_basic_eda(df)

    print_step("Save Plots")
    outputs = [
        save_target_count_plot(df),
        save_numeric_histograms(df),
        save_credit_amount_boxplot(df),
        save_correlation_heatmap(df),
        save_purpose_default_rate_plot(df),
    ]

    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
