"""
Deterministic data-cleaning utilities for the EMIPredict AI project.

Responsibilities
----------------
This module performs dataset-level deterministic cleaning:

1. Numeric type normalization.
2. Categorical whitespace normalization.
3. Duplicate-row detection and removal.
4. Missing-value reporting.
5. Domain-violation detection.
6. Conversion of domain-invalid numerical values to NaN.

This module intentionally does NOT perform learned imputation.

Imputation, scaling, and categorical encoding belong to the
training-fitted preprocessing pipeline in:

    src/preprocessing/preprocessing_pipeline.py

This separation prevents preprocessing leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS: Final[tuple[str, ...]] = (
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
)


NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
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
)


CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
    "emi_eligibility",
)


DOMAIN_RANGES: Final[dict[str, tuple[float | None, float | None]]] = {
    "age": (25, 60),
    "monthly_salary": (15_000, 200_000),
    "years_of_employment": (0, None),
    "family_size": (1, None),
    "dependents": (0, None),
    "credit_score": (300, 850),
    "requested_amount": (0, None),
    "requested_tenure": (1, None),
    "max_monthly_emi": (500, None),
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_columns(dataframe: pd.DataFrame) -> None:
    """
    Validate that the dataset contains the expected columns.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.

    ValueError
        If required columns are missing.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )


# ---------------------------------------------------------------------------
# Numeric normalization
# ---------------------------------------------------------------------------


def normalize_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert expected numeric columns to numeric dtype.

    Values that cannot be parsed are converted to NaN.

    Examples
    --------
    ``"58.0"`` becomes ``58.0``.

    Malformed values such as ``"58.0.0"`` become ``NaN``.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized numeric columns.
    """
    cleaned = dataframe.copy()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    return cleaned


# ---------------------------------------------------------------------------
# Categorical normalization
# ---------------------------------------------------------------------------


def clean_categorical_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize whitespace in categorical columns.

    Missing values remain missing.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized categorical values.
    """
    cleaned = dataframe.copy()

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].map(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else value
            )
        )

    return cleaned


# ---------------------------------------------------------------------------
# Duplicate analysis
# ---------------------------------------------------------------------------


def calculate_duplicate_report(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """
    Calculate duplicate-row statistics.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    dict[str, int]
        Duplicate statistics.
    """
    duplicate_mask = dataframe.duplicated(
        keep=False
    )

    duplicate_row_count = int(
        duplicate_mask.sum()
    )

    duplicate_group_count = int(
        dataframe.loc[
            duplicate_mask
        ].drop_duplicates().shape[0]
    )

    return {
        "total_rows": int(len(dataframe)),
        "duplicate_rows": duplicate_row_count,
        "duplicate_groups": duplicate_group_count,
    }


def remove_exact_duplicates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove exact duplicate rows.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe without exact duplicates.
    """
    before = len(dataframe)

    cleaned = dataframe.drop_duplicates(
        keep="first"
    ).copy()

    removed = before - len(cleaned)

    logger.info(
        "Removed %d exact duplicate rows.",
        removed,
    )

    return cleaned


# ---------------------------------------------------------------------------
# Missing-value reporting
# ---------------------------------------------------------------------------


def calculate_missing_value_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a column-level missing-value report.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Missing-value summary.
    """
    total_rows = len(dataframe)

    records: list[dict[str, float | int | str]] = []

    for column in dataframe.columns:
        missing_count = int(
            dataframe[column].isna().sum()
        )

        missing_percentage = (
            missing_count / total_rows * 100
            if total_rows > 0
            else 0.0
        )

        records.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "non_missing_count": (
                    total_rows - missing_count
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by="missing_count",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Domain validation
# ---------------------------------------------------------------------------


def detect_domain_violations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect values outside documented domain ranges.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Domain-violation summary.
    """
    records: list[dict[str, float | int | str]] = []

    for column, bounds in DOMAIN_RANGES.items():
        minimum, maximum = bounds

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        valid_mask = values.notna()

        below_minimum = (
            (values < minimum).fillna(False).sum()
            if minimum is not None
            else 0
        )

        above_maximum = (
            (values > maximum).fillna(False).sum()
            if maximum is not None
            else 0
        )

        total_violations = (
            int(below_minimum)
            + int(above_maximum)
        )

        valid_count = int(
            valid_mask.sum()
        )

        violation_percentage = (
            total_violations
            / valid_count
            * 100
            if valid_count > 0
            else 0.0
        )

        records.append(
            {
                "column": column,
                "domain_minimum": minimum,
                "domain_maximum": maximum,
                "valid_numeric_count": valid_count,
                "below_minimum_count": int(
                    below_minimum
                ),
                "above_maximum_count": int(
                    above_maximum
                ),
                "total_violation_count": (
                    total_violations
                ),
                "violation_percentage": (
                    violation_percentage
                ),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Domain-invalid value handling
# ---------------------------------------------------------------------------


def convert_domain_invalid_values_to_nan(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert known domain-invalid numerical values to NaN.

    The affected rows are preserved.

    This function does NOT perform imputation.

    Domain rules
    ------------
    Values outside DOMAIN_RANGES are considered invalid.

    For example:

    - credit_score < 300
    - credit_score > 850
    - monthly_salary < 15000
    - monthly_salary > 200000

    are converted to NaN.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with invalid domain values replaced by NaN.
    """
    cleaned = dataframe.copy()

    total_replacements = 0

    for column, bounds in DOMAIN_RANGES.items():
        minimum, maximum = bounds

        values = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

        invalid_mask = pd.Series(
            False,
            index=cleaned.index,
        )

        if minimum is not None:
            invalid_mask |= values < minimum

        if maximum is not None:
            invalid_mask |= values > maximum

        replacement_count = int(
            invalid_mask.sum()
        )

        if replacement_count > 0:
            cleaned.loc[
                invalid_mask,
                column,
            ] = pd.NA

            total_replacements += (
                replacement_count
            )

            logger.warning(
                "Converted %d invalid values "
                "to NaN in '%s'.",
                replacement_count,
                column,
            )

    logger.info(
        "Total domain-invalid values converted "
        "to NaN: %d",
        total_replacements,
    )

    return cleaned


# ---------------------------------------------------------------------------
# Complete cleaning workflow
# ---------------------------------------------------------------------------


def clean_dataset(
    dataframe: pd.DataFrame,
    remove_duplicates: bool = True,
    convert_invalid_domain_values: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | dict[str, int]]]:
    """
    Run deterministic dataset cleaning.

    Parameters
    ----------
    dataframe:
        Raw input dataframe.

    remove_duplicates:
        Whether to remove exact duplicate rows.

    convert_invalid_domain_values:
        Whether domain-invalid numerical values should be
        converted to NaN.

    Returns
    -------
    tuple
        Cleaned dataframe and audit reports.

    Notes
    -----
    No statistical imputation is performed here.

    Imputation belongs to the training-fitted preprocessing
    pipeline.
    """
    validate_columns(dataframe)

    logger.info(
        "Starting deterministic data cleaning."
    )

    cleaned = dataframe.copy()

    duplicate_report = (
        calculate_duplicate_report(cleaned)
    )

    if remove_duplicates:
        cleaned = remove_exact_duplicates(
            cleaned
        )

    cleaned = normalize_numeric_columns(
        cleaned
    )

    cleaned = clean_categorical_values(
        cleaned
    )

    # Capture domain violations BEFORE replacing
    # invalid values with NaN.
    domain_report = detect_domain_violations(
        cleaned
    )

    if convert_invalid_domain_values:
        cleaned = (
            convert_domain_invalid_values_to_nan(
                cleaned
            )
        )

    missing_report = (
        calculate_missing_value_report(cleaned)
    )

    reports: dict[
        str,
        pd.DataFrame | dict[str, int],
    ] = {
        "duplicate_summary": duplicate_report,
        "domain_violations": domain_report,
        "missing_values": missing_report,
    }

    logger.info(
        "Deterministic data cleaning completed."
    )

    logger.info(
        "Final dataset shape: %s",
        cleaned.shape,
    )

    return cleaned, reports


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_cleaning_reports(
    reports: dict[
        str,
        pd.DataFrame | dict[str, int],
    ],
    output_directory: str | Path,
) -> None:
    """
    Save cleaning reports to disk.

    Parameters
    ----------
    reports:
        Reports generated by clean_dataset().

    output_directory:
        Destination directory.
    """
    output_path = Path(
        output_directory
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for report_name, report in reports.items():
        destination = (
            output_path / f"{report_name}.csv"
        )

        if isinstance(report, pd.DataFrame):
            report.to_csv(
                destination,
                index=False,
            )

        elif isinstance(report, dict):
            pd.DataFrame(
                [report]
            ).to_csv(
                destination,
                index=False,
            )

        else:
            raise TypeError(
                f"Unsupported report type for "
                f"'{report_name}': "
                f"{type(report).__name__}"
            )

        logger.info(
            "Saved cleaning report: %s",
            destination,
        )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args() -> object:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Perform deterministic cleaning of "
            "the raw EMI prediction dataset."
        )
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the raw CSV dataset.",
    )

    parser.add_argument(
        "--output-directory",
        required=True,
        help=(
            "Directory where cleaning reports "
            "will be saved."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run deterministic cleaning from the command line.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    args = parse_args()

    data_path = Path(args.data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    logger.info(
        "Loading dataset: %s",
        data_path,
    )

    dataframe = pd.read_csv(
        data_path,
        low_memory=False,
    )

    logger.info(
        "Dataset loaded successfully: %s",
        dataframe.shape,
    )

    cleaned_dataframe, reports = clean_dataset(
        dataframe=dataframe,
        remove_duplicates=True,
        convert_invalid_domain_values=True,
    )

    save_cleaning_reports(
        reports=reports,
        output_directory=args.output_directory,
    )

    logger.info(
        "Cleaned dataset shape: %s",
        cleaned_dataframe.shape,
    )

    logger.info(
        "Cleaning completed successfully."
    )


if __name__ == "__main__":
    main()