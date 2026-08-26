"""
Module: data_validation.py

Description:
    Data-quality validation utilities for the EMIPredict AI project.

    This module validates the structure and quality of the EMI dataset
    before the cleaning and feature-engineering stages.

    The module does not modify the input DataFrame.

Author:
    EMIPredict AI Project
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


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

EXPECTED_TARGET_COLUMNS = [
    "emi_eligibility",
    "max_monthly_emi",
]

NUMERICAL_COLUMNS = [
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


def validate_structure(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate the structural properties of the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    dict[str, Any]
        Structural validation report.
    """
    actual_columns = dataframe.columns.tolist()

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

    target_columns_present = all(
        column in dataframe.columns
        for column in EXPECTED_TARGET_COLUMNS
    )

    return {
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "target_columns_present": target_columns_present,
        "column_order_matches": actual_columns == EXPECTED_COLUMNS,
    }


def check_missing_values(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a missing-value report.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to inspect.

    Returns
    -------
    pd.DataFrame
        Missing-value report containing count and percentage.
    """
    missing_count = dataframe.isna().sum()

    missing_report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": (
                missing_count / len(dataframe) * 100
                if len(dataframe) > 0
                else 0.0
            ),
        }
    )

    missing_report = missing_report.sort_values(
        by="missing_count",
        ascending=False,
    )

    return missing_report


def check_duplicates(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check for duplicate records.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to inspect.

    Returns
    -------
    dict[str, Any]
        Duplicate-record summary.
    """
    duplicate_count = int(dataframe.duplicated().sum())

    duplicate_percentage = (
        duplicate_count / len(dataframe) * 100
        if len(dataframe) > 0
        else 0.0
    )

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
    }


def check_data_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a data-type report.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to inspect.

    Returns
    -------
    pd.DataFrame
        Data-type and uniqueness report.
    """
    report = pd.DataFrame(
        {
            "dtype": dataframe.dtypes.astype(str),
            "non_null_count": dataframe.notna().sum(),
            "unique_count": dataframe.nunique(dropna=True),
        }
    )

    report["expected_category"] = "unknown"

    numerical_mask = report.index.isin(NUMERICAL_COLUMNS)
    categorical_mask = report.index.isin(CATEGORICAL_COLUMNS)

    report.loc[numerical_mask, "expected_category"] = "numerical"
    report.loc[categorical_mask, "expected_category"] = "categorical"

    return report


def check_domain_constraints(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check basic domain constraints specified by the project document.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    dict[str, Any]
        Domain validation results.

    Notes
    -----
    The project specification provides expected ranges for age,
    monthly salary, credit score, and max monthly EMI. It also
    defines the classification target categories.
    """
    violations: dict[str, int] = {}

    if "age" in dataframe.columns:
        age_num = pd.to_numeric(dataframe["age"], errors="coerce")
        violations["age_out_of_range"] = int(
            (
                (age_num < 25)
                | (age_num > 60)
            ).sum()
        )

    if "monthly_salary" in dataframe.columns:
        salary_num = pd.to_numeric(dataframe["monthly_salary"], errors="coerce")
        violations["salary_out_of_range"] = int(
            (
                (salary_num < 15000)
                | (salary_num > 200000)
            ).sum()
        )

    if "credit_score" in dataframe.columns:
        credit_num = pd.to_numeric(dataframe["credit_score"], errors="coerce")
        violations["credit_score_out_of_range"] = int(
            (
                (credit_num < 300)
                | (credit_num > 850)
            ).sum()
        )

    if "max_monthly_emi" in dataframe.columns:
        emi_num = pd.to_numeric(dataframe["max_monthly_emi"], errors="coerce")
        violations["max_monthly_emi_out_of_range"] = int(
            (
                (emi_num < 500)
                | (emi_num > 50000)
            ).sum()
        )

    if "emi_eligibility" in dataframe.columns:
        expected_classes = {
            "Eligible",
            "High_Risk",
            "Not_Eligible",
        }

        observed_classes = set(
            dataframe["emi_eligibility"]
            .dropna()
            .astype(str)
            .unique()
        )

        unexpected_classes = sorted(
            observed_classes - expected_classes
        )

        violations["unexpected_eligibility_classes"] = len(
            unexpected_classes
        )

    return violations


def validate_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run all dataset validation checks.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    dict[str, Any]
        Complete validation report.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    logger.info("Starting dataset validation.")

    structure_report = validate_structure(dataframe)
    missing_report = check_missing_values(dataframe)
    duplicate_report = check_duplicates(dataframe)
    dtype_report = check_data_types(dataframe)
    domain_report = check_domain_constraints(dataframe)

    validation_report = {
        "structure": structure_report,
        "missing_values": missing_report,
        "duplicates": duplicate_report,
        "data_types": dtype_report,
        "domain_constraints": domain_report,
    }

    logger.info("Dataset validation completed.")

    return validation_report


if __name__ == "__main__":
    from load_data import load_dataset

    DATA_PATH = "data/raw/emi_prediction_dataset.csv"

    dataset = load_dataset(DATA_PATH)

    report = validate_dataset(dataset)

    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print("\nStructure:")
    for key, value in report["structure"].items():
        print(f"{key}: {value}")

    print("\nMissing Values:")
    print(report["missing_values"])

    print("\nDuplicates:")
    for key, value in report["duplicates"].items():
        print(f"{key}: {value}")

    print("\nData Types:")
    print(report["data_types"])

    print("\nDomain Constraints:")
    for key, value in report["domain_constraints"].items():
        print(f"{key}: {value}")