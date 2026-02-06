import os
from pathlib import Path

MODELING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODELING_DIR.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
MPLCONFIG_DIR = CACHE_DIR / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = PROJECT_ROOT / "data/accepted/accepted_2007_to_2018Q4.csv"
REJECTED_DATA_PATH = PROJECT_ROOT / "data/rejected/rejected_2007_to_2018Q4.csv"
FIGURES_DIR = MODELING_DIR / "figures"
TARGET_COLUMN = "default_flag"
OUTCOME_COLUMN = "target_label"
STATUS_COLUMN = "loan_status"
PLOT_SAMPLE_SIZE = 100_000
REJECTED_SAMPLE_SIZE = 100_000

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

READ_COLUMNS = [
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
]

NUMERIC_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

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
}

REJECTED_READ_COLUMNS = [
    "Amount Requested",
    "Application Date",
    "Loan Title",
    "Risk_Score",
    "Debt-To-Income Ratio",
    "State",
    "Employment Length",
]

REJECTED_READ_DTYPE = {
    "Amount Requested": "float32",
    "Loan Title": "category",
    "Risk_Score": "float32",
    "Debt-To-Income Ratio": "string",
    "State": "category",
    "Employment Length": "category",
}


def print_step(name: str) -> None:
    print(f"\n=== {name} ===")


def clean_percent_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float32")

    cleaned = series.astype("string").str.strip().str.rstrip("%")
    return pd.to_numeric(cleaned, errors="coerce").astype("float32")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    df = pd.read_csv(path, usecols=READ_COLUMNS, dtype=READ_DTYPE)
    raw_rows = len(df)

    df["int_rate"] = clean_percent_column(df["int_rate"])
    df["revol_util"] = clean_percent_column(df["revol_util"])
    df["term_months"] = (
        df["term"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False)
        .astype("float32")
    )
    df["fico_score"] = (
            (df["fico_range_low"] + df["fico_range_high"]) / 2
    ).astype("float32")

    df[TARGET_COLUMN] = pd.NA
    df[OUTCOME_COLUMN] = pd.NA

    default_mask = df[STATUS_COLUMN].isin(DEFAULT_STATUSES)
    non_default_mask = df[STATUS_COLUMN].isin(NON_DEFAULT_STATUSES)
    df.loc[default_mask, TARGET_COLUMN] = 1
    df.loc[default_mask, OUTCOME_COLUMN] = "Default"
    df.loc[non_default_mask, TARGET_COLUMN] = 0
    df.loc[non_default_mask, OUTCOME_COLUMN] = "Non-Default"

    model_df = df[df[TARGET_COLUMN].notna()].copy()
    model_df[TARGET_COLUMN] = model_df[TARGET_COLUMN].astype("int8")
    model_df[OUTCOME_COLUMN] = model_df[OUTCOME_COLUMN].astype("category")
    for column in [STATUS_COLUMN, OUTCOME_COLUMN, *CATEGORICAL_COLUMNS]:
        if isinstance(model_df[column].dtype, pd.CategoricalDtype):
            model_df[column] = model_df[column].cat.remove_unused_categories()
    model_df.attrs["raw_rows"] = raw_rows
    model_df.attrs["excluded_rows"] = raw_rows - len(model_df)
    return model_df


def load_rejected_data(path: Path = REJECTED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing rejected data file: {path}")

    df = pd.read_csv(path, usecols=REJECTED_READ_COLUMNS, dtype=REJECTED_READ_DTYPE)
    df = df.rename(
        columns={
            "Amount Requested": "amount_requested",
            "Application Date": "application_date",
            "Loan Title": "loan_title",
            "Risk_Score": "risk_score",
            "Debt-To-Income Ratio": "rejected_dti",
            "State": "state",
            "Employment Length": "employment_length",
        }
    )
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df["application_year"] = df["application_date"].dt.year.astype("float32")
    df["rejected_dti"] = clean_percent_column(df["rejected_dti"])
    return df


def sample_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= PLOT_SAMPLE_SIZE:
        return df
    return df.sample(PLOT_SAMPLE_SIZE, random_state=42)


def sample_rejected_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= REJECTED_SAMPLE_SIZE:
        return df
    return df.sample(REJECTED_SAMPLE_SIZE, random_state=42)


def show_basic_eda(df: pd.DataFrame) -> None:
    raw_rows = df.attrs.get("raw_rows", len(df))
    excluded_rows = df.attrs.get("excluded_rows", 0)

    print("Raw rows:", raw_rows)
    print("Model-ready shape:", df.shape)
    print("Excluded rows without final/default outcome:", excluded_rows)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df[FEATURE_COLUMNS + [STATUS_COLUMN, TARGET_COLUMN, OUTCOME_COLUMN]].head())
    print("\nNumeric summary:")
    print(df[NUMERIC_COLUMNS].describe().T)
    print("\nLoan status count:")
    print(df[STATUS_COLUMN].value_counts())
    print("\nTarget count:")
    print(df[OUTCOME_COLUMN].value_counts())
    print(f"\nDefault rate: {df[TARGET_COLUMN].mean():.2%}")
    print("\nMissing values:")
    missing = df[FEATURE_COLUMNS].isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing if not missing.empty else "No missing values found.")


def show_rejected_eda(df: pd.DataFrame) -> None:
    print("Rejected shape:", df.shape)
    print("\nRejected preview:")
    print(df.head())
    print("\nRejected numeric summary:")
    print(df[["amount_requested", "risk_score", "rejected_dti"]].describe().T)
    print("\nTop rejected states:")
    print(df["state"].value_counts().head(10))


def save_target_count_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "target_count.png"

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=OUTCOME_COLUMN, order=["Non-Default", "Default"])
    plt.title("LendingClub Target Count")
    plt.xlabel("Loan Outcome")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_numeric_histograms(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "numeric_distributions.png"
    plot_df = sample_for_plot(df)

    plot_df[NUMERIC_COLUMNS].hist(figsize=(14, 10), bins=30)
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_loan_amount_boxplot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "loan_amount_by_target.png"
    plot_df = sample_for_plot(df)

    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=plot_df,
        x=OUTCOME_COLUMN,
        y="loan_amnt",
        order=["Non-Default", "Default"],
    )
    plt.title("Loan Amount by Loan Outcome")
    plt.xlabel("Loan Outcome")
    plt.ylabel("Loan Amount")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_correlation_heatmap(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "numeric_correlation_heatmap.png"
    plot_df = sample_for_plot(df)

    corr_columns = [
        "loan_amnt",
        "annual_inc",
        "dti",
        "fico_score",
        "revol_util",
        "int_rate",
        "term_months",
        TARGET_COLUMN,
    ]
    corr = plot_df[corr_columns].corr()
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
        df.groupby("purpose", observed=True, as_index=False)[TARGET_COLUMN]
        .mean()
        .sort_values(TARGET_COLUMN, ascending=False)
    )

    plt.figure(figsize=(9, 4))
    sns.barplot(data=purpose_default, x="purpose", y=TARGET_COLUMN)
    plt.title("Default Rate by Purpose")
    plt.xlabel("Purpose")
    plt.ylabel("Default Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_grade_default_rate_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "grade_default_rate.png"

    grade_default = (
        df.groupby("grade", observed=True, as_index=False)[TARGET_COLUMN]
        .mean()
        .sort_values("grade")
    )

    plt.figure(figsize=(8, 4))
    sns.barplot(data=grade_default, x="grade", y=TARGET_COLUMN, color="#4C78A8")
    plt.title("Default Rate by Credit Grade")
    plt.xlabel("Credit Grade")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_grade_interest_boxplot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "interest_rate_by_grade.png"
    plot_df = sample_for_plot(df)

    plt.figure(figsize=(9, 4))
    sns.boxplot(data=plot_df, x="grade", y="int_rate", color="#72B7B2", showfliers=False)
    plt.title("Interest Rate Distribution by Credit Grade")
    plt.xlabel("Credit Grade")
    plt.ylabel("Interest Rate (%)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_fico_dti_by_target_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fico_dti_by_target.png"
    plot_df = sample_for_plot(df)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df,
        x="fico_score",
        y="dti",
        hue=OUTCOME_COLUMN,
        hue_order=["Non-Default", "Default"],
        alpha=0.25,
        s=14,
        linewidth=0,
    )
    plt.title("FICO Score and DTI by Loan Outcome")
    plt.xlabel("FICO Score")
    plt.ylabel("Debt-to-Income Ratio")
    plt.legend(title="Loan Outcome")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_fico_default_rate_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "fico_band_default_rate.png"
    plot_df = df[["fico_score", TARGET_COLUMN]].dropna().copy()
    plot_df["fico_band"] = pd.cut(
        plot_df["fico_score"],
        bins=[300, 580, 620, 660, 700, 740, 780, 850],
        labels=["300-579", "580-619", "620-659", "660-699", "700-739", "740-779", "780-850"],
        include_lowest=True,
    )
    fico_default = (
        plot_df.groupby("fico_band", observed=True, as_index=False)[TARGET_COLUMN]
        .mean()
        .dropna()
    )

    plt.figure(figsize=(9, 4))
    sns.lineplot(data=fico_default, x="fico_band", y=TARGET_COLUMN, marker="o")
    plt.title("Default Rate by FICO Band")
    plt.xlabel("FICO Band")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_revol_util_violin_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "revol_util_by_target.png"
    plot_df = sample_for_plot(df)

    plt.figure(figsize=(7, 4))
    sns.violinplot(
        data=plot_df,
        x=OUTCOME_COLUMN,
        y="revol_util",
        order=["Non-Default", "Default"],
        inner="quartile",
        cut=0,
    )
    plt.title("Revolving Utilization by Loan Outcome")
    plt.xlabel("Loan Outcome")
    plt.ylabel("Revolving Utilization (%)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_income_loan_heatmap(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "income_loan_amount_default_heatmap.png"
    plot_df = df[["annual_inc", "loan_amnt", TARGET_COLUMN]].dropna().copy()
    plot_df["annual_inc"] = plot_df["annual_inc"].clip(upper=250_000)
    plot_df["income_band"] = pd.qcut(plot_df["annual_inc"], q=5, duplicates="drop")
    plot_df["loan_amount_band"] = pd.qcut(plot_df["loan_amnt"], q=5, duplicates="drop")
    heatmap_data = plot_df.pivot_table(
        index="income_band",
        columns="loan_amount_band",
        values=TARGET_COLUMN,
        aggfunc="mean",
        observed=True,
    )

    plt.figure(figsize=(10, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap="YlOrRd")
    plt.title("Default Rate by Income and Loan Amount Bands")
    plt.xlabel("Loan Amount Band")
    plt.ylabel("Annual Income Band")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_term_grade_default_rate_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "term_grade_default_rate.png"
    rate_table = df.pivot_table(
        index="grade",
        columns="term_months",
        values=TARGET_COLUMN,
        aggfunc="mean",
        observed=True,
    ).sort_index()

    plt.figure(figsize=(7, 4))
    sns.heatmap(rate_table, annot=True, fmt=".1%", cmap="YlOrRd")
    plt.title("Default Rate by Grade and Term")
    plt.xlabel("Term (Months)")
    plt.ylabel("Credit Grade")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_rejected_risk_score_histogram(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "rejected_risk_score_distribution.png"
    plot_df = sample_rejected_for_plot(df).dropna(subset=["risk_score"])

    plt.figure(figsize=(8, 4))
    sns.histplot(data=plot_df, x="risk_score", bins=40, kde=True, color="#F58518")
    plt.title("Rejected Applications Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_rejected_amount_dti_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "rejected_amount_vs_dti.png"
    plot_df = sample_rejected_for_plot(df).dropna(
        subset=["amount_requested", "rejected_dti", "risk_score"]
    )
    plot_df = plot_df[plot_df["rejected_dti"].between(0, 100)]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df,
        x="amount_requested",
        y="rejected_dti",
        hue="risk_score",
        palette="viridis",
        alpha=0.3,
        s=12,
        linewidth=0,
    )
    plt.title("Rejected Amount Requested vs DTI")
    plt.xlabel("Amount Requested")
    plt.ylabel("Debt-to-Income Ratio (%)")
    plt.legend(title="Risk Score")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_rejected_state_count_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "top_rejected_states.png"
    state_counts = df["state"].value_counts().head(15).rename_axis("state").reset_index(name="count")

    plt.figure(figsize=(9, 4))
    sns.barplot(data=state_counts, x="state", y="count", color="#54A24B")
    plt.title("Top Rejected Application States")
    plt.xlabel("State")
    plt.ylabel("Rejected Applications")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def save_rejected_yearly_trend_plot(df: pd.DataFrame) -> Path:
    out = FIGURES_DIR / "rejected_applications_by_year.png"
    yearly_counts = (
        df.dropna(subset=["application_year"])
        .groupby("application_year", as_index=False)
        .size()
    )

    plt.figure(figsize=(9, 4))
    sns.lineplot(data=yearly_counts, x="application_year", y="size", marker="o")
    plt.title("Rejected Applications by Year")
    plt.xlabel("Application Year")
    plt.ylabel("Rejected Applications")
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
    rejected_df = load_rejected_data()
    print(f"Loaded from: {REJECTED_DATA_PATH}")

    print_step("Basic EDA")
    show_basic_eda(df)
    print_step("Rejected EDA")
    show_rejected_eda(rejected_df)

    print_step("Save Plots")
    outputs = [
        save_target_count_plot(df),
        save_numeric_histograms(df),
        save_loan_amount_boxplot(df),
        save_correlation_heatmap(df),
        save_purpose_default_rate_plot(df),
        save_grade_default_rate_plot(df),
        save_grade_interest_boxplot(df),
        save_fico_dti_by_target_plot(df),
        save_fico_default_rate_plot(df),
        save_revol_util_violin_plot(df),
        save_income_loan_heatmap(df),
        save_term_grade_default_rate_plot(df),
        save_rejected_risk_score_histogram(rejected_df),
        save_rejected_amount_dti_plot(rejected_df),
        save_rejected_state_count_plot(rejected_df),
        save_rejected_yearly_trend_plot(rejected_df),
    ]

    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
