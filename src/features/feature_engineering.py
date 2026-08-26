"""
Feature engineering for the EMIPredict AI project.

This module creates deterministic, domain-interpretable financial
features. It does not perform imputation, encoding, scaling, feature
selection, or model fitting.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "monthly_salary",
    "monthly_rent",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "requested_amount",
    "requested_tenure",
)


GENERATED_FEATURES: Final[tuple[str, ...]] = (
    "total_education_expenses",
    "total_monthly_living_expenses",
    "total_monthly_expenses",
    "disposable_income",
    "expense_to_income_ratio",
    "emi_to_income_ratio",
    "requested_amount_to_income_ratio",
    "requested_amount_per_month",
)


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """
    Validate that all required source columns exist.

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
        raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required feature-engineering columns: "
            f"{missing_columns}"
        )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Perform element-wise division safely.

    Zero denominators produce NaN rather than infinity.

    Parameters
    ----------
    numerator:
        Numerator values.

    denominator:
        Denominator values.

    Returns
    -------
    pd.Series
        Division results with invalid divisions represented as NaN.
    """
    numerator_numeric = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator_numeric = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype="float64",
    )

    valid_denominator = (
        denominator_numeric.notna()
        & denominator_numeric.ne(0)
    )

    result.loc[valid_denominator] = (
        numerator_numeric.loc[valid_denominator]
        / denominator_numeric.loc[valid_denominator]
    )

    return result


def create_expense_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create aggregate monthly expense features.

    Missing source values are preserved rather than silently treated
    as zero. This keeps feature engineering separate from imputation.

    Parameters
    ----------
    dataframe:
        Input dataframe containing required financial columns.

    Returns
    -------
    pd.DataFrame
        Dataframe containing expense-derived features.
    """
    education_columns = [
        "school_fees",
        "college_fees",
    ]

    living_expense_columns = [
        "monthly_rent",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
    ]

    monthly_expense_columns = [
        "total_education_expenses",
        "total_monthly_living_expenses",
        "current_emi_amount",
    ]

    dataframe["total_education_expenses"] = dataframe[
        education_columns
    ].sum(axis=1, min_count=len(education_columns))

    dataframe["total_monthly_living_expenses"] = dataframe[
        living_expense_columns
    ].sum(axis=1, min_count=len(living_expense_columns))

    dataframe["total_monthly_expenses"] = dataframe[
        monthly_expense_columns
    ].sum(axis=1, min_count=len(monthly_expense_columns))

    monthly_salary_numeric = pd.to_numeric(
        dataframe["monthly_salary"],
        errors="coerce",
    )

    total_expenses_numeric = pd.to_numeric(
        dataframe["total_monthly_expenses"],
        errors="coerce",
    )

    dataframe["disposable_income"] = (
        monthly_salary_numeric - total_expenses_numeric
    )

    return dataframe


def create_ratio_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create income and financial-burden ratio features.

    Parameters
    ----------
    dataframe:
        Dataframe containing engineered expense features.

    Returns
    -------
    pd.DataFrame
        Dataframe containing ratio-based features.
    """
    dataframe["expense_to_income_ratio"] = safe_divide(
        dataframe["total_monthly_expenses"],
        dataframe["monthly_salary"],
    )

    dataframe["emi_to_income_ratio"] = safe_divide(
        dataframe["current_emi_amount"],
        dataframe["monthly_salary"],
    )

    dataframe["requested_amount_to_income_ratio"] = safe_divide(
        dataframe["requested_amount"],
        dataframe["monthly_salary"],
    )

    dataframe["requested_amount_per_month"] = safe_divide(
        dataframe["requested_amount"],
        dataframe["requested_tenure"],
    )

    return dataframe


def engineer_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate deterministic financial features.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with engineered features.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.

    ValueError
        If required source columns are missing.
    """
    validate_required_columns(dataframe)

    engineered_data = dataframe.copy()

    engineered_data = create_expense_features(
        engineered_data
    )

    engineered_data = create_ratio_features(
        engineered_data
    )

    logger.info(
        "Feature engineering completed successfully."
    )

    logger.info(
        "Original feature count: %d",
        len(dataframe.columns),
    )

    logger.info(
        "Final feature count: %d",
        len(engineered_data.columns),
    )

    logger.info(
        "Generated features: %s",
        list(GENERATED_FEATURES),
    )

    return engineered_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    raise SystemExit(
        "feature_engineering.py is a library module. "
        "Use engineer_features() from another module."
    )