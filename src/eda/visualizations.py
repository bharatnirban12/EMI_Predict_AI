"""
Visual exploratory data analysis for the EMI prediction dataset.

Important:
This module intentionally operates on the raw dataset.

It does not:
- impute missing values,
- correct malformed numeric values,
- remove outliers,
- normalize categories,
- perform feature engineering,
- train models.

The purpose is to visually inspect the raw data before
designing the data-cleaning and modeling pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOGGER = logging.getLogger(__name__)


CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"


NUMERIC_COLUMNS = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
    "max_monthly_emi",
]


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - %(message)s"
        ),
    )


def load_dataset(
    data_path: Path,
) -> pd.DataFrame:
    """Load the raw EMI dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    dataframe = pd.read_csv(
        data_path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Dataset is empty."
        )

    LOGGER.info(
        "Loaded dataset: %s",
        dataframe.shape,
    )

    return dataframe


def validate_columns(
    dataframe: pd.DataFrame,
) -> None:
    """Validate columns required for visualization."""
    required_columns = (
        NUMERIC_COLUMNS
        + [
            CLASSIFICATION_TARGET,
            "emi_scenario",
            "existing_loans",
            "employment_type",
            "house_type",
            "education",
            "gender",
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def save_figure(
    figure: plt.Figure,
    output_path: Path,
) -> None:
    """Save and close a matplotlib figure."""
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    LOGGER.info(
        "Saved figure: %s",
        output_path,
    )


def plot_target_distribution(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot classification-target distribution."""
    counts = (
        dataframe[CLASSIFICATION_TARGET]
        .fillna("<MISSING>")
        .value_counts()
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    counts.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "EMI Eligibility Distribution"
    )
    axis.set_xlabel(
        "EMI Eligibility"
    )
    axis.set_ylabel(
        "Record Count"
    )

    save_figure(
        figure,
        output_directory
        / "01_emi_eligibility_distribution.png",
    )


def plot_numeric_distribution(
    dataframe: pd.DataFrame,
    column: str,
    filename: str,
    output_directory: Path,
) -> None:
    """
    Plot a raw numerical-variable distribution.

    Conversion is performed only for plotting so malformed
    numeric strings remain visible as missing/unparseable
    observations rather than being silently corrected.
    """
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.hist(
        values.dropna(),
        bins=50,
    )

    axis.set_title(
        f"Distribution of {column}"
    )
    axis.set_xlabel(column)
    axis.set_ylabel("Frequency")

    save_figure(
        figure,
        output_directory / filename,
    )


def plot_categorical_target_relationship(
    dataframe: pd.DataFrame,
    column: str,
    filename: str,
    output_directory: Path,
) -> None:
    """Plot classification proportions within each category."""
    working = dataframe[
        [column, CLASSIFICATION_TARGET]
    ].copy()

    working[column] = working[column].fillna(
        "<MISSING>"
    )

    proportions = pd.crosstab(
        working[column],
        working[CLASSIFICATION_TARGET],
        normalize="index",
    ) * 100

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    proportions.plot(
        kind="bar",
        stacked=True,
        ax=axis,
    )

    axis.set_title(
        f"{column} vs EMI Eligibility"
    )
    axis.set_xlabel(column)
    axis.set_ylabel(
        "Percentage Within Category"
    )

    axis.legend(
        title=CLASSIFICATION_TARGET
    )

    save_figure(
        figure,
        output_directory / filename,
    )


def plot_numeric_vs_classification(
    dataframe: pd.DataFrame,
    column: str,
    filename: str,
    output_directory: Path,
) -> None:
    """Plot numerical distribution by classification class."""
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    working = pd.DataFrame(
        {
            "value": values,
            CLASSIFICATION_TARGET: (
                dataframe[
                    CLASSIFICATION_TARGET
                ]
            ),
        }
    ).dropna()

    groups = []

    labels = []

    for target_class in [
        "Eligible",
        "High_Risk",
        "Not_Eligible",
    ]:
        subset = working.loc[
            working[CLASSIFICATION_TARGET]
            == target_class,
            "value",
        ]

        if not subset.empty:
            groups.append(subset)
            labels.append(target_class)

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.boxplot(
        groups,
        tick_labels=labels,
        showfliers=False,
    )

    axis.set_title(
        f"{column} by EMI Eligibility"
    )
    axis.set_xlabel(
        "EMI Eligibility"
    )
    axis.set_ylabel(column)

    save_figure(
        figure,
        output_directory / filename,
    )


def plot_numeric_vs_regression(
    dataframe: pd.DataFrame,
    feature: str,
    filename: str,
    output_directory: Path,
) -> None:
    """Plot numerical feature against regression target."""
    x = pd.to_numeric(
        dataframe[feature],
        errors="coerce",
    )

    y = pd.to_numeric(
        dataframe[REGRESSION_TARGET],
        errors="coerce",
    )

    valid = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    # Plot a sample for readability when the dataset is large.
    if len(valid) > 50_000:
        valid = valid.sample(
            n=50_000,
            random_state=42,
        )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.scatter(
        valid["x"],
        valid["y"],
        alpha=0.15,
        s=8,
    )

    axis.set_title(
        f"{feature} vs {REGRESSION_TARGET}"
    )
    axis.set_xlabel(feature)
    axis.set_ylabel(REGRESSION_TARGET)

    save_figure(
        figure,
        output_directory / filename,
    )


def plot_correlation_heatmap(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot numeric Pearson correlation matrix."""
    numeric_data = pd.DataFrame()

    for column in NUMERIC_COLUMNS:
        numeric_data[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    correlation = numeric_data.corr()

    figure, axis = plt.subplots(
        figsize=(15, 12)
    )

    image = axis.imshow(
        correlation,
        aspect="auto",
    )

    axis.set_xticks(
        range(len(correlation.columns))
    )

    axis.set_yticks(
        range(len(correlation.columns))
    )

    axis.set_xticklabels(
        correlation.columns,
        rotation=90,
    )

    axis.set_yticklabels(
        correlation.columns
    )

    axis.set_title(
        "Numeric Feature Correlation Matrix"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Pearson Correlation",
    )

    save_figure(
        figure,
        output_directory
        / "27_numeric_correlation_heatmap.png",
    )


def plot_missing_values(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot missing-value percentages."""
    missing_percentage = (
        dataframe.isna()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    missing_percentage = (
        missing_percentage[
            missing_percentage > 0
        ]
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    if missing_percentage.empty:
        axis.text(
            0.5,
            0.5,
            "No missing values detected",
            ha="center",
            va="center",
        )
        axis.set_axis_off()
    else:
        missing_percentage.plot(
            kind="bar",
            ax=axis,
        )

        axis.set_title(
            "Missing Values by Column"
        )
        axis.set_xlabel("Column")
        axis.set_ylabel(
            "Missing Percentage"
        )

    save_figure(
        figure,
        output_directory
        / "28_missing_values.png",
    )


def run_visual_analysis(
    data_path: Path,
    output_directory: Path,
) -> None:
    """Run complete visual EDA."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_dataset(
        data_path
    )

    validate_columns(
        dataframe
    )

    plot_target_distribution(
        dataframe,
        output_directory,
    )

    numeric_distribution_specs = [
        (
            "age",
            "02_age_distribution.png",
        ),
        (
            "monthly_salary",
            "03_monthly_salary_distribution.png",
        ),
        (
            "credit_score",
            "04_credit_score_distribution.png",
        ),
        (
            "bank_balance",
            "05_bank_balance_distribution.png",
        ),
        (
            "emergency_fund",
            "06_emergency_fund_distribution.png",
        ),
        (
            "requested_amount",
            "07_requested_amount_distribution.png",
        ),
        (
            "requested_tenure",
            "08_requested_tenure_distribution.png",
        ),
        (
            "max_monthly_emi",
            "09_max_monthly_emi_distribution.png",
        ),
    ]

    for column, filename in numeric_distribution_specs:
        plot_numeric_distribution(
            dataframe,
            column,
            filename,
            output_directory,
        )

    categorical_specs = [
        (
            "emi_scenario",
            "10_scenario_vs_eligibility.png",
        ),
        (
            "existing_loans",
            "11_existing_loans_vs_eligibility.png",
        ),
        (
            "employment_type",
            "12_employment_vs_eligibility.png",
        ),
        (
            "house_type",
            "13_house_type_vs_eligibility.png",
        ),
        (
            "education",
            "14_education_vs_eligibility.png",
        ),
        (
            "gender",
            "15_gender_vs_eligibility.png",
        ),
    ]

    for column, filename in categorical_specs:
        plot_categorical_target_relationship(
            dataframe,
            column,
            filename,
            output_directory,
        )

    classification_specs = [
        (
            "monthly_salary",
            "16_salary_vs_eligibility.png",
        ),
        (
            "credit_score",
            "17_credit_score_vs_eligibility.png",
        ),
        (
            "bank_balance",
            "18_bank_balance_vs_eligibility.png",
        ),
        (
            "current_emi_amount",
            "19_current_emi_vs_eligibility.png",
        ),
        (
            "requested_amount",
            "20_requested_amount_vs_eligibility.png",
        ),
    ]

    for column, filename in classification_specs:
        plot_numeric_vs_classification(
            dataframe,
            column,
            filename,
            output_directory,
        )

    regression_specs = [
        (
            "monthly_salary",
            "21_salary_vs_max_emi.png",
        ),
        (
            "bank_balance",
            "22_bank_balance_vs_max_emi.png",
        ),
        (
            "groceries_utilities",
            "23_groceries_vs_max_emi.png",
        ),
        (
            "travel_expenses",
            "24_travel_expenses_vs_max_emi.png",
        ),
        (
            "emergency_fund",
            "25_emergency_fund_vs_max_emi.png",
        ),
        (
            "current_emi_amount",
            "26_current_emi_vs_max_emi.png",
        ),
    ]

    for feature, filename in regression_specs:
        plot_numeric_vs_regression(
            dataframe,
            feature,
            filename,
            output_directory,
        )

    plot_correlation_heatmap(
        dataframe,
        output_directory,
    )

    plot_missing_values(
        dataframe,
        output_directory,
    )

    LOGGER.info(
        "Visual EDA completed successfully."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate visual EDA for the raw EMI "
            "prediction dataset."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()

    arguments = parse_arguments()

    run_visual_analysis(
        data_path=arguments.data_path,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()