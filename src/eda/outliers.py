"""
Outlier and extreme-value analysis for the raw EMI dataset.

This module performs analysis only.

It does NOT:
    - remove observations,
    - cap observations,
    - impute missing values,
    - modify the source dataset,
    - perform feature engineering,
    - train models.

The analysis distinguishes between:

1. Statistical outliers
2. Domain-bound violations
3. Malformed numeric values

These categories must not be treated as equivalent.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


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


DOMAIN_BOUNDS = {
    "age": {
        "minimum": 25,
        "maximum": 60,
    },
    "monthly_salary": {
        "minimum": 15_000,
        "maximum": 200_000,
    },
    "years_of_employment": {
        "minimum": 0,
        "maximum": None,
    },
    "family_size": {
        "minimum": 1,
        "maximum": None,
    },
    "dependents": {
        "minimum": 0,
        "maximum": None,
    },
    "credit_score": {
        "minimum": 300,
        "maximum": 850,
    },
    "requested_amount": {
        "minimum": 0,
        "maximum": None,
    },
    "requested_tenure": {
        "minimum": 1,
        "maximum": None,
    },
    "max_monthly_emi": {
        "minimum": 500,
        "maximum": None,
    },
}


PERCENTILES = [
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99,
]


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - %(message)s"
        ),
    )


def load_dataset(
    data_path: Path,
) -> pd.DataFrame:
    """
    Load the raw CSV dataset.

    Parameters
    ----------
    data_path:
        Path to the raw dataset.

    Returns
    -------
    pandas.DataFrame
        Raw dataset.
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


def validate_columns(
    dataframe: pd.DataFrame,
) -> None:
    """Validate required numerical columns."""
    missing_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


def calculate_outlier_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate IQR-based outlier statistics.

    Numeric conversion is performed with errors='coerce'.
    Malformed non-null values are counted separately.
    """
    records = []

    for column in NUMERIC_COLUMNS:
        raw_series = dataframe[column]

        numeric_series = pd.to_numeric(
            raw_series,
            errors="coerce",
        )

        raw_non_null_count = int(
            raw_series.notna().sum()
        )

        numeric_count = int(
            numeric_series.notna().sum()
        )

        malformed_count = (
            raw_non_null_count
            - numeric_count
        )

        if numeric_count == 0:
            records.append(
                {
                    "feature": column,
                    "raw_non_null_count": (
                        raw_non_null_count
                    ),
                    "numeric_count": numeric_count,
                    "malformed_numeric_count": (
                        malformed_count
                    ),
                    "q1": None,
                    "q3": None,
                    "iqr": None,
                    "iqr_lower_bound": None,
                    "iqr_upper_bound": None,
                    "iqr_outlier_count": 0,
                    "iqr_outlier_percentage": 0.0,
                }
            )
            continue

        q1 = float(
            numeric_series.quantile(0.25)
        )

        q3 = float(
            numeric_series.quantile(0.75)
        )

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outlier_mask = (
            (numeric_series < lower_bound)
            | (numeric_series > upper_bound)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        outlier_percentage = (
            outlier_count
            / numeric_count
            * 100
        )

        records.append(
            {
                "feature": column,
                "raw_non_null_count": (
                    raw_non_null_count
                ),
                "numeric_count": numeric_count,
                "malformed_numeric_count": (
                    malformed_count
                ),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "iqr_lower_bound": lower_bound,
                "iqr_upper_bound": upper_bound,
                "iqr_outlier_count": (
                    outlier_count
                ),
                "iqr_outlier_percentage": (
                    outlier_percentage
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_percentile_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate distribution percentiles for numerical features.
    """
    records = []

    for column in NUMERIC_COLUMNS:
        numeric_series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna()

        if numeric_series.empty:
            continue

        values = numeric_series.quantile(
            PERCENTILES
        )

        records.append(
            {
                "feature": column,
                "count": int(
                    numeric_series.count()
                ),
                "minimum": float(
                    numeric_series.min()
                ),
                "p01": float(
                    values.loc[0.01]
                ),
                "p05": float(
                    values.loc[0.05]
                ),
                "p25": float(
                    values.loc[0.25]
                ),
                "p50": float(
                    values.loc[0.50]
                ),
                "p75": float(
                    values.loc[0.75]
                ),
                "p95": float(
                    values.loc[0.95]
                ),
                "p99": float(
                    values.loc[0.99]
                ),
                "maximum": float(
                    numeric_series.max()
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_domain_outlier_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate domain-bound violations.

    Domain violations are reported separately from IQR
    statistical outliers.
    """
    records = []

    for column, bounds in DOMAIN_BOUNDS.items():
        numeric_series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        valid_count = int(
            numeric_series.notna().sum()
        )

        minimum = bounds["minimum"]
        maximum = bounds["maximum"]

        if minimum is None:
            below_minimum_count = 0
        else:
            below_minimum_count = int(
                (
                    numeric_series < minimum
                ).sum()
            )

        if maximum is None:
            above_maximum_count = 0
        else:
            above_maximum_count = int(
                (
                    numeric_series > maximum
                ).sum()
            )

        total_violation_count = (
            below_minimum_count
            + above_maximum_count
        )

        violation_percentage = (
            (
                total_violation_count
                / valid_count
                * 100
            )
            if valid_count
            else 0.0
        )

        records.append(
            {
                "feature": column,
                "domain_minimum": minimum,
                "domain_maximum": maximum,
                "valid_numeric_count": valid_count,
                "below_minimum_count": (
                    below_minimum_count
                ),
                "above_maximum_count": (
                    above_maximum_count
                ),
                "total_violation_count": (
                    total_violation_count
                ),
                "violation_percentage": (
                    violation_percentage
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_extreme_value_examples(
    dataframe: pd.DataFrame,
    maximum_examples_per_feature: int = 10,
) -> pd.DataFrame:
    """
    Extract extreme numeric examples.

    The output is an investigation aid only.

    For each feature, the examples include:
        - smallest numeric observations
        - largest numeric observations

    Malformed strings are handled separately and are not
    silently converted.
    """
    records = []

    for column in NUMERIC_COLUMNS:
        numeric_series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        numeric_indices = numeric_series.dropna()

        if numeric_indices.empty:
            continue

        smallest = numeric_indices.nsmallest(
            maximum_examples_per_feature
        )

        largest = numeric_indices.nlargest(
            maximum_examples_per_feature
        )

        for row_index, value in smallest.items():
            records.append(
                {
                    "feature": column,
                    "direction": "smallest",
                    "row_index": int(row_index),
                    "numeric_value": float(value),
                    "raw_value": str(
                        dataframe.loc[
                            row_index,
                            column,
                        ]
                    ),
                }
            )

        for row_index, value in largest.items():
            records.append(
                {
                    "feature": column,
                    "direction": "largest",
                    "row_index": int(row_index),
                    "numeric_value": float(value),
                    "raw_value": str(
                        dataframe.loc[
                            row_index,
                            column,
                        ]
                    ),
                }
            )

    return pd.DataFrame(records)


def save_outputs(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Generate and save all outlier-analysis tables."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    outlier_summary = (
        calculate_outlier_summary(
            dataframe
        )
    )

    percentile_summary = (
        calculate_percentile_summary(
            dataframe
        )
    )

    domain_summary = (
        calculate_domain_outlier_summary(
            dataframe
        )
    )

    extreme_examples = (
        calculate_extreme_value_examples(
            dataframe
        )
    )

    outlier_summary.to_csv(
        output_directory
        / "outlier_summary.csv",
        index=False,
    )

    percentile_summary.to_csv(
        output_directory
        / "percentile_summary.csv",
        index=False,
    )

    domain_summary.to_csv(
        output_directory
        / "domain_outlier_summary.csv",
        index=False,
    )

    extreme_examples.to_csv(
        output_directory
        / "extreme_value_examples.csv",
        index=False,
    )

    LOGGER.info(
        "Saved outlier summary."
    )

    LOGGER.info(
        "Saved percentile summary."
    )

    LOGGER.info(
        "Saved domain outlier summary."
    )

    LOGGER.info(
        "Saved extreme-value examples."
    )


def run_outlier_analysis(
    data_path: Path,
    output_directory: Path,
) -> None:
    """Run complete outlier analysis."""
    dataframe = load_dataset(
        data_path
    )

    validate_columns(
        dataframe
    )

    save_outputs(
        dataframe,
        output_directory,
    )

    LOGGER.info(
        "Outlier analysis completed successfully."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Perform statistical and domain outlier "
            "analysis on the raw EMI dataset."
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
        help="Directory for EDA tables.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging()

    arguments = parse_arguments()

    run_outlier_analysis(
        data_path=arguments.data_path,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()