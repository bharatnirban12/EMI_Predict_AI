"""
Data-quality analysis for the raw EMI prediction dataset.

This module is diagnostic only.

It identifies:
- missing values,
- malformed numeric values,
- categorical inconsistencies,
- domain-bound violations,
- duplicate rows.

It does not:
- modify the raw dataset,
- impute values,
- remove records,
- normalize categories,
- perform feature engineering.
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


CATEGORICAL_COLUMNS = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
    "emi_eligibility",
]


# These are diagnostic bounds only.
#
# They are NOT automatically cleaning rules.
#
# The final project requirements/documentation must be
# consulted before treating any bound as a hard business rule.
DOMAIN_BOUNDS = {
    "age": {
        "minimum": 25,
        "maximum": 60,
    },
    "monthly_salary": {
        "minimum": 15000,
        "maximum": 200000,
    },
    "years_of_employment": {
        "minimum": 0,
    },
    "family_size": {
        "minimum": 1,
    },
    "dependents": {
        "minimum": 0,
    },
    "credit_score": {
        "minimum": 300,
        "maximum": 850,
    },
    "requested_amount": {
        "minimum": 0,
    },
    "requested_tenure": {
        "minimum": 1,
    },
    "max_monthly_emi": {
        "minimum": 500,
    },
}


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
    Load the raw dataset without modifying its values.
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


def analyze_missing_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate missing-value statistics for every column.
    """
    result = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_count": [
                int(dataframe[column].isna().sum())
                for column in dataframe.columns
            ],
        }
    )

    result["missing_percentage"] = (
        result["missing_count"]
        / len(dataframe)
        * 100
    )

    result["non_missing_count"] = (
        len(dataframe)
        - result["missing_count"]
    )

    return result.sort_values(
        "missing_count",
        ascending=False,
    ).reset_index(drop=True)


def analyze_numeric_parseability(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine whether numeric-looking columns can be parsed
    as numeric values.

    The original values are preserved.
    """
    records = []

    for column in NUMERIC_COLUMNS:
        if column not in dataframe.columns:
            continue

        series = dataframe[column]

        parsed = pd.to_numeric(
            series,
            errors="coerce",
        )

        original_non_null = int(
            series.notna().sum()
        )

        parsed_non_null = int(
            parsed.notna().sum()
        )

        unparseable_count = (
            original_non_null
            - parsed_non_null
        )

        records.append(
            {
                "column": column,
                "raw_dtype": str(series.dtype),
                "total_rows": len(series),
                "raw_missing_count": int(
                    series.isna().sum()
                ),
                "raw_non_missing_count": (
                    original_non_null
                ),
                "numeric_parseable_count": (
                    parsed_non_null
                ),
                "unparseable_non_null_count": (
                    unparseable_count
                ),
                "unparseable_percentage_of_rows": (
                    unparseable_count
                    / len(series)
                    * 100
                ),
            }
        )

    return pd.DataFrame(records)


def collect_unparseable_examples(
    dataframe: pd.DataFrame,
    max_examples: int = 20,
) -> pd.DataFrame:
    """
    Collect examples of non-null values that cannot be parsed
    numerically.
    """
    records = []

    for column in NUMERIC_COLUMNS:
        if column not in dataframe.columns:
            continue

        series = dataframe[column]

        parsed = pd.to_numeric(
            series,
            errors="coerce",
        )

        mask = (
            series.notna()
            & parsed.isna()
        )

        examples = series.loc[mask].head(
            max_examples
        )

        for index, value in examples.items():
            records.append(
                {
                    "column": column,
                    "row_index": int(index),
                    "raw_value": str(value),
                }
            )

    return pd.DataFrame(records)


def analyze_categorical_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate frequency distributions for categorical columns.
    """
    records = []

    for column in CATEGORICAL_COLUMNS:
        if column not in dataframe.columns:
            continue

        counts = (
            dataframe[column]
            .value_counts(
                dropna=False
            )
        )

        for value, count in counts.items():
            if pd.isna(value):
                category = "<MISSING>"
            else:
                category = str(value)

            records.append(
                {
                    "column": column,
                    "value": category,
                    "count": int(count),
                    "percentage": (
                        float(
                            count
                            / len(dataframe)
                            * 100
                        )
                    ),
                }
            )

    return pd.DataFrame(records)


def analyze_domain_violations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify numeric values outside configured diagnostic bounds.

    Bounds are reported only. No rows are removed.
    """
    records = []

    for column, bounds in DOMAIN_BOUNDS.items():
        if column not in dataframe.columns:
            continue

        numeric_series = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if "minimum" in bounds:
            minimum = bounds["minimum"]

            below_mask = (
                numeric_series.notna()
                & (
                    numeric_series
                    < minimum
                )
            )

            below_count = int(
                below_mask.sum()
            )
        else:
            minimum = None
            below_count = 0

        if "maximum" in bounds:
            maximum = bounds["maximum"]

            above_mask = (
                numeric_series.notna()
                & (
                    numeric_series
                    > maximum
                )
            )

            above_count = int(
                above_mask.sum()
            )
        else:
            maximum = None
            above_count = 0

        records.append(
            {
                "column": column,
                "minimum_bound": minimum,
                "maximum_bound": maximum,
                "below_minimum_count": (
                    below_count
                ),
                "above_maximum_count": (
                    above_count
                ),
                "total_violation_count": (
                    below_count
                    + above_count
                ),
            }
        )

    return pd.DataFrame(records)


def analyze_duplicates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize exact duplicate rows.
    """
    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    return pd.DataFrame(
        [
            {
                "metric": "total_rows",
                "value": len(dataframe),
            },
            {
                "metric": "duplicate_rows",
                "value": duplicate_count,
            },
            {
                "metric": "duplicate_percentage",
                "value": (
                    duplicate_count
                    / len(dataframe)
                    * 100
                ),
            },
        ]
    )


def save_outputs(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Run all quality analyses and save their outputs.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing_values = (
        analyze_missing_values(dataframe)
    )

    numeric_parseability = (
        analyze_numeric_parseability(
            dataframe
        )
    )

    unparseable_examples = (
        collect_unparseable_examples(
            dataframe
        )
    )

    categorical_values = (
        analyze_categorical_values(
            dataframe
        )
    )

    domain_violations = (
        analyze_domain_violations(
            dataframe
        )
    )

    duplicate_summary = (
        analyze_duplicates(dataframe)
    )

    missing_values.to_csv(
        output_directory
        / "missing_values.csv",
        index=False,
    )

    numeric_parseability.to_csv(
        output_directory
        / "numeric_parseability.csv",
        index=False,
    )

    unparseable_examples.to_csv(
        output_directory
        / "unparseable_numeric_examples.csv",
        index=False,
    )

    categorical_values.to_csv(
        output_directory
        / "categorical_values.csv",
        index=False,
    )

    domain_violations.to_csv(
        output_directory
        / "domain_violations.csv",
        index=False,
    )

    duplicate_summary.to_csv(
        output_directory
        / "duplicate_summary.csv",
        index=False,
    )

    LOGGER.info(
        "Saved data-quality outputs to: %s",
        output_directory,
    )


def run_data_quality_analysis(
    data_path: Path,
    output_directory: Path,
) -> None:
    """
    Execute the complete data-quality analysis.
    """
    dataframe = load_dataset(data_path)

    save_outputs(
        dataframe=dataframe,
        output_directory=output_directory,
    )

    LOGGER.info(
        "Data-quality analysis completed."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze raw EMI dataset quality."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to raw CSV.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for quality outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the data-quality analysis CLI."""
    configure_logging()

    arguments = parse_arguments()

    run_data_quality_analysis(
        data_path=arguments.data_path,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()