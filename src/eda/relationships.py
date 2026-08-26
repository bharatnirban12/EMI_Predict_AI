"""
Relationship analysis for the EMI prediction dataset.

This module performs exploratory relationship analysis only.

It does not:
- clean the raw dataset,
- impute missing values,
- remove outliers,
- encode categorical variables,
- create model features,
- train models.

Outputs include:
- categorical variable vs classification target distributions,
- numerical variable vs classification target summaries,
- numerical variable vs regression target summaries,
- numerical correlation matrix.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"


CATEGORICAL_ANALYSIS_COLUMNS = [
    "emi_scenario",
    "gender",
    "education",
    "employment_type",
    "house_type",
    "existing_loans",
]


NUMERIC_ANALYSIS_COLUMNS = [
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
]


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - %(message)s"
        ),
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load the raw dataset.

    Parameters
    ----------
    data_path:
        Path to the raw CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded raw dataset.
    """
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
            "The dataset is empty."
        )

    LOGGER.info(
        "Loaded dataset: %s",
        dataframe.shape,
    )

    return dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate columns required by this analysis.
    """
    required_columns = (
        CATEGORICAL_ANALYSIS_COLUMNS
        + NUMERIC_ANALYSIS_COLUMNS
        + [
            CLASSIFICATION_TARGET,
            REGRESSION_TARGET,
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


def calculate_categorical_target_distribution(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Calculate target distribution within each category.

    Percentages are calculated within each category, not
    across the entire dataset.

    This allows direct comparison such as:

        Government:
            Eligible %
            High_Risk %
            Not_Eligible %

        Private:
            Eligible %
            High_Risk %
            Not_Eligible %
    """
    working = dataframe[
        [column, CLASSIFICATION_TARGET]
    ].copy()

    working[column] = working[column].fillna(
        "<MISSING>"
    )

    counts = (
        working.groupby(
            [column, CLASSIFICATION_TARGET],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    category_totals = (
        counts.groupby(column)["count"]
        .transform("sum")
    )

    counts["percentage_within_category"] = (
        counts["count"]
        / category_totals
        * 100
    )

    counts = counts.sort_values(
        [column, CLASSIFICATION_TARGET]
    ).reset_index(drop=True)

    return counts


def calculate_all_categorical_distributions(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Calculate classification distributions for all
    configured categorical variables.
    """
    results = {}

    for column in CATEGORICAL_ANALYSIS_COLUMNS:
        LOGGER.info(
            "Analyzing %s × %s",
            column,
            CLASSIFICATION_TARGET,
        )

        results[column] = (
            calculate_categorical_target_distribution(
                dataframe,
                column,
            )
        )

    return results


def calculate_numeric_classification_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate numerical-variable summaries by classification.
    """
    records = []

    for column in NUMERIC_ANALYSIS_COLUMNS:
        numeric_series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        working = pd.DataFrame(
            {
                "value": numeric_series,
                CLASSIFICATION_TARGET: (
                    dataframe[
                        CLASSIFICATION_TARGET
                    ]
                ),
            }
        )

        grouped = (
            working.groupby(
                CLASSIFICATION_TARGET,
                dropna=False,
            )["value"]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                ]
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            records.append(
                {
                    "feature": column,
                    "target_class": row[
                        CLASSIFICATION_TARGET
                    ],
                    "count": int(
                        row["count"]
                    ),
                    "mean": float(
                        row["mean"]
                    ),
                    "median": float(
                        row["median"]
                    ),
                    "std": float(
                        row["std"]
                    ),
                    "min": float(
                        row["min"]
                    ),
                    "max": float(
                        row["max"]
                    ),
                }
            )

    return pd.DataFrame(records)


def calculate_numeric_regression_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate relationships between numerical predictors
    and the regression target.
    """
    records = []

    target = pd.to_numeric(
        dataframe[REGRESSION_TARGET],
        errors="coerce",
    )

    for column in NUMERIC_ANALYSIS_COLUMNS:
        feature = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        valid_mask = (
            feature.notna()
            & target.notna()
        )

        valid_feature = feature.loc[
            valid_mask
        ]

        valid_target = target.loc[
            valid_mask
        ]

        if valid_mask.sum() == 0:
            correlation = float("nan")
        elif valid_feature.nunique() <= 1:
            correlation = float("nan")
        else:
            correlation = float(
                valid_feature.corr(
                    valid_target
                )
            )

        records.append(
            {
                "feature": column,
                "valid_pair_count": int(
                    valid_mask.sum()
                ),
                "feature_mean": float(
                    valid_feature.mean()
                ),
                "feature_median": float(
                    valid_feature.median()
                ),
                "target_mean": float(
                    valid_target.mean()
                ),
                "target_median": float(
                    valid_target.median()
                ),
                "pearson_correlation": (
                    correlation
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        "pearson_correlation",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)


def calculate_correlation_matrix(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the Pearson correlation matrix for numeric
    variables and the regression target.

    Object columns are not coerced here. The purpose is to
    analyze the numeric variables that are currently
    interpretable as numeric.
    """
    columns = (
        NUMERIC_ANALYSIS_COLUMNS
        + [REGRESSION_TARGET]
    )

    numeric_dataframe = pd.DataFrame()

    for column in columns:
        numeric_dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return numeric_dataframe.corr(
        method="pearson"
    )


def save_categorical_outputs(
    categorical_results: dict[str, pd.DataFrame],
    output_directory: Path,
) -> None:
    """
    Save categorical relationship tables.
    """
    file_mapping = {
        "emi_scenario": (
            "scenario_target_distribution.csv"
        ),
        "gender": (
            "gender_target_distribution.csv"
        ),
        "education": (
            "education_target_distribution.csv"
        ),
        "employment_type": (
            "employment_target_distribution.csv"
        ),
        "house_type": (
            "house_type_target_distribution.csv"
        ),
        "existing_loans": (
            "existing_loans_target_distribution.csv"
        ),
    }

    for column, dataframe in categorical_results.items():
        output_path = (
            output_directory
            / file_mapping[column]
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        LOGGER.info(
            "Saved: %s",
            output_path,
        )


def save_outputs(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Execute and save all relationship analyses.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    categorical_results = (
        calculate_all_categorical_distributions(
            dataframe
        )
    )

    numeric_classification_summary = (
        calculate_numeric_classification_summary(
            dataframe
        )
    )

    numeric_regression_summary = (
        calculate_numeric_regression_summary(
            dataframe
        )
    )

    correlation_matrix = (
        calculate_correlation_matrix(
            dataframe
        )
    )

    save_categorical_outputs(
        categorical_results,
        output_directory,
    )

    numeric_classification_summary.to_csv(
        output_directory
        / "numeric_classification_summary.csv",
        index=False,
    )

    numeric_regression_summary.to_csv(
        output_directory
        / "numeric_regression_summary.csv",
        index=False,
    )

    correlation_matrix.to_csv(
        output_directory
        / "correlation_matrix.csv"
    )

    LOGGER.info(
        "Saved numerical classification summary."
    )

    LOGGER.info(
        "Saved numerical regression summary."
    )

    LOGGER.info(
        "Saved correlation matrix."
    )


def run_relationship_analysis(
    data_path: Path,
    output_directory: Path,
) -> None:
    """
    Run the complete relationship-analysis workflow.
    """
    dataframe = load_dataset(data_path)

    validate_required_columns(
        dataframe
    )

    save_outputs(
        dataframe=dataframe,
        output_directory=output_directory,
    )

    LOGGER.info(
        "Relationship analysis completed."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze relationships in the raw EMI "
            "prediction dataset."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the raw CSV dataset.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for relationship outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the relationship-analysis CLI."""
    configure_logging()

    arguments = parse_arguments()

    run_relationship_analysis(
        data_path=arguments.data_path,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()