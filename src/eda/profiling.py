"""
Dataset profiling utilities for the EMI prediction project.

This module performs descriptive profiling only.

It does not:
- modify the raw dataset,
- impute missing values,
- remove records,
- encode categorical variables,
- perform feature engineering,
- train models.

The purpose is to establish a reproducible factual description
of the raw dataset before any cleaning or modeling decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)


EXPECTED_COLUMNS = [
    "age",
    "gender",
    "marital_status",
    "education",
    "monthly_salary",
    "employment_type",
    "years_of_employment",
    "company_type",
    "house_type",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "existing_loans",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "emi_scenario",
    "requested_amount",
    "requested_tenure",
    "emi_eligibility",
    "max_monthly_emi",
]

CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"

EXPECTED_INPUT_COLUMNS = [
    column
    for column in EXPECTED_COLUMNS
    if column not in {
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET,
    }
]


def configure_logging() -> None:
    """Configure console logging for the profiling module."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - %(message)s"
        ),
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load the raw EMI dataset.

    Parameters
    ----------
    data_path:
        Path to the raw CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    ValueError
        If the CSV is empty.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    if not data_path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {data_path}"
        )

    LOGGER.info("Loading dataset: %s", data_path)

    dataframe = pd.read_csv(
        data_path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "The dataset is empty."
        )

    LOGGER.info(
        "Dataset loaded successfully: %s",
        dataframe.shape,
    )

    return dataframe


def validate_schema(dataframe: pd.DataFrame) -> None:
    """
    Validate the raw dataset schema.

    Parameters
    ----------
    dataframe:
        Dataset to validate.

    Raises
    ------
    ValueError
        If required columns are missing or unexpected
        columns are present.
    """
    actual_columns = list(dataframe.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if unexpected_columns:
        raise ValueError(
            "Unexpected columns found: "
            f"{unexpected_columns}"
        )

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Column order does not match the expected "
            "raw dataset schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )

    LOGGER.info(
        "Schema validation passed: %d columns.",
        len(actual_columns),
    )


def build_column_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive information for every column.

    Returns
    -------
    pandas.DataFrame
        Column-level profile.
    """
    records: list[dict[str, Any]] = []

    total_rows = len(dataframe)

    for column in dataframe.columns:
        series = dataframe[column]

        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "row_count": total_rows,
                "non_null_count": int(
                    series.notna().sum()
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_percentage": float(
                    series.isna().mean() * 100
                ),
                "unique_count": int(
                    series.nunique(dropna=True)
                ),
                "duplicate_value_count": int(
                    total_rows
                    - series.nunique(dropna=False)
                ),
            }
        )

    return pd.DataFrame(records)


def build_numeric_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive statistics for columns that pandas
    currently recognizes as numeric.

    This intentionally does not coerce object columns.
    """
    numeric_dataframe = dataframe.select_dtypes(
        include="number"
    )

    if numeric_dataframe.empty:
        return pd.DataFrame()

    profile = (
        numeric_dataframe
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "column"})
    )

    return profile


def build_categorical_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build cardinality information for object/category columns.
    """
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category"]
    ).columns

    records: list[dict[str, Any]] = []

    for column in categorical_columns:
        series = dataframe[column]

        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "unique_count": int(
                    series.nunique(dropna=True)
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
            }
        )

    return pd.DataFrame(records)


def build_target_profile(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build descriptive information for both project targets.
    """
    classification_series = dataframe[
        CLASSIFICATION_TARGET
    ]

    regression_series = dataframe[
        REGRESSION_TARGET
    ]

    classification_distribution = (
        classification_series
        .value_counts(dropna=False)
        .rename_axis("class")
        .reset_index(name="count")
    )

    classification_distribution[
        "percentage"
    ] = (
        classification_distribution["count"]
        / len(dataframe)
        * 100
    )

    regression_summary = (
        regression_series
        .describe()
        .to_dict()
    )

    return {
        "classification_target": (
            CLASSIFICATION_TARGET
        ),
        "classification_distribution": (
            classification_distribution.to_dict(
                orient="records"
            )
        ),
        "regression_target": REGRESSION_TARGET,
        "regression_summary": regression_summary,
    }


def build_dataset_profile(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build the complete raw dataset profile.
    """
    return {
        "dataset": {
            "row_count": int(len(dataframe)),
            "column_count": int(
                len(dataframe.columns)
            ),
            "duplicate_row_count": int(
                dataframe.duplicated().sum()
            ),
        },
        "schema": {
            "columns": list(dataframe.columns),
            "input_columns": EXPECTED_INPUT_COLUMNS,
            "classification_target": (
                CLASSIFICATION_TARGET
            ),
            "regression_target": REGRESSION_TARGET,
        },
        "targets": build_target_profile(
            dataframe
        ),
    }


def save_profile_outputs(
    dataframe: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Save profiling outputs to disk.
    """
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    column_profile = build_column_profile(
        dataframe
    )

    numeric_profile = build_numeric_profile(
        dataframe
    )

    categorical_profile = (
        build_categorical_profile(dataframe)
    )

    dataset_profile = build_dataset_profile(
        dataframe
    )

    column_profile_path = (
        output_directory
        / "column_profile.csv"
    )

    numeric_profile_path = (
        output_directory
        / "numeric_profile.csv"
    )

    categorical_profile_path = (
        output_directory
        / "categorical_profile.csv"
    )

    dataset_profile_path = (
        output_directory
        / "dataset_profile.json"
    )

    column_profile.to_csv(
        column_profile_path,
        index=False,
    )

    if not numeric_profile.empty:
        numeric_profile.to_csv(
            numeric_profile_path,
            index=False,
        )

    if not categorical_profile.empty:
        categorical_profile.to_csv(
            categorical_profile_path,
            index=False,
        )

    with dataset_profile_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            dataset_profile,
            file,
            indent=4,
            default=str,
        )

    LOGGER.info(
        "Saved column profile: %s",
        column_profile_path,
    )

    LOGGER.info(
        "Saved numeric profile: %s",
        numeric_profile_path,
    )

    LOGGER.info(
        "Saved categorical profile: %s",
        categorical_profile_path,
    )

    LOGGER.info(
        "Saved dataset profile: %s",
        dataset_profile_path,
    )


def run_profiling(
    data_path: Path,
    output_directory: Path,
) -> None:
    """
    Execute the complete raw-dataset profiling workflow.
    """
    dataframe = load_dataset(data_path)

    validate_schema(dataframe)

    LOGGER.info(
        "Rows: %d",
        len(dataframe),
    )

    LOGGER.info(
        "Columns: %d",
        len(dataframe.columns),
    )

    LOGGER.info(
        "Duplicate rows: %d",
        dataframe.duplicated().sum(),
    )

    save_profile_outputs(
        dataframe=dataframe,
        output_directory=output_directory,
    )

    LOGGER.info(
        "Raw dataset profiling completed successfully."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile the raw EMI prediction dataset."
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
        help="Directory for profiling outputs.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the profiling command-line interface."""
    configure_logging()

    arguments = parse_arguments()

    run_profiling(
        data_path=arguments.data_path,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()