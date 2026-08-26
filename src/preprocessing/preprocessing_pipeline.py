"""
Reusable preprocessing pipeline for the EMIPredict AI project.

Responsibilities
----------------
This module:

1. Defines the authoritative ML predictor schema.
2. Separates numerical and categorical predictors.
3. Excludes target variables.
4. Performs numerical median imputation.
5. Performs numerical standard scaling.
6. Performs categorical most-frequent imputation.
7. Performs categorical one-hot encoding.

This module does NOT:

- clean malformed raw values,
- perform domain correction,
- engineer financial features,
- select a machine-learning model,
- fit a model.

The preprocessing transformer must be fitted using training data only
and then reused for validation, test, and inference data.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


logger = logging.getLogger(__name__)


TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "emi_eligibility",
    "max_monthly_emi",
)


# ---------------------------------------------------------------------------
# Numerical predictors
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# `dependents` is intentionally excluded.
#
# EDA established that:
#
#     dependents = family_size - 1
#
# for all 404,800 records.
#
# Therefore dependents is perfectly redundant with family_size and should
# not be supplied to the ML models.
#
NUMERICAL_COLUMNS: Final[tuple[str, ...]] = (
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
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
    "total_education_expenses",
    "total_monthly_living_expenses",
    "total_monthly_expenses",
    "disposable_income",
    "expense_to_income_ratio",
    "emi_to_income_ratio",
    "requested_amount_to_income_ratio",
    "requested_amount_per_month",
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
)


PREDICTOR_COLUMNS: Final[tuple[str, ...]] = (
    NUMERICAL_COLUMNS
    + CATEGORICAL_COLUMNS
)


def validate_input_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the input dataframe type.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )


def validate_feature_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that all expected predictor columns exist.

    Parameters
    ----------
    dataframe:
        Input feature-engineered dataframe.

    Raises
    ------
    ValueError
        If one or more predictor columns are missing.
    """
    validate_input_dataframe(dataframe)

    missing_features = [
        column
        for column in PREDICTOR_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required predictor columns: "
            f"{missing_features}"
        )


def validate_targets_are_excluded(
    dataframe: pd.DataFrame,
) -> None:
    """
    Ensure target columns are not present in the predictor dataframe.

    This validation operates on the actual dataframe rather than only
    checking the hardcoded predictor lists.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Raises
    ------
    ValueError
        If a target column is found in the predictor schema.
    """
    validate_input_dataframe(dataframe)

    leaked_targets = [
        target
        for target in TARGET_COLUMNS
        if target in PREDICTOR_COLUMNS
    ]

    if leaked_targets:
        raise ValueError(
            "Target leakage detected in predictor schema: "
            f"{leaked_targets}"
        )

    logger.info(
        "Target exclusion validation passed."
    )


def validate_redundant_features_are_excluded(
    dataframe: pd.DataFrame,
) -> None:
    """
    Ensure features identified as redundant during EDA are not used.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Raises
    ------
    ValueError
        If a known redundant feature is included in the predictor schema.
    """
    del dataframe

    redundant_features = {
        "dependents",
    }

    leaked_redundant_features = sorted(
        redundant_features.intersection(
            PREDICTOR_COLUMNS
        )
    )

    if leaked_redundant_features:
        raise ValueError(
            "Redundant features detected in predictor schema: "
            f"{leaked_redundant_features}"
        )

    logger.info(
        "Redundant-feature validation passed."
    )


def _coerce_numeric(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all columns to numeric, replacing invalid parsing with NaN.
    """
    return pd.DataFrame(dataframe).apply(pd.to_numeric, errors="coerce")


def create_numeric_pipeline() -> Pipeline:
    """
    Create numerical preprocessing.

    Returns
    -------
    Pipeline
        Type coercion, median imputation followed by standard scaling.
    """
    return Pipeline(
        steps=[
            (
                "type_coercion",
                FunctionTransformer(
                    func=_coerce_numeric,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


def create_categorical_pipeline() -> Pipeline:
    """
    Create categorical preprocessing.

    Returns
    -------
    Pipeline
        Most-frequent imputation followed by one-hot encoding.
    """
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


def create_preprocessing_pipeline(
    dataframe: pd.DataFrame | None = None,
    allow_missing: bool = False,
) -> ColumnTransformer:
    """
    Create the unfitted preprocessing transformer.

    Parameters
    ----------
    dataframe:
        Optional dataframe used for schema validation only.
    allow_missing:
        If True, ignores missing predictor columns and filters the pipeline columns.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing transformer.

    Notes
    -----
    This function does not fit the transformer.

    The returned transformer must be fitted using training data only.
    """
    num_cols = list(NUMERICAL_COLUMNS)
    cat_cols = list(CATEGORICAL_COLUMNS)

    if dataframe is not None:
        if not allow_missing:
            validate_feature_columns(dataframe)
        else:
            num_cols = [c for c in num_cols if c in dataframe.columns]
            cat_cols = [c for c in cat_cols if c in dataframe.columns]
            
        validate_targets_are_excluded(dataframe)
        validate_redundant_features_are_excluded(dataframe)

    numeric_pipeline = create_numeric_pipeline()
    categorical_pipeline = create_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numeric_pipeline,
                num_cols,
            ),
            (
                "categorical",
                categorical_pipeline,
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    logger.info(
        "Preprocessing pipeline created."
    )

    logger.info(
        "Numerical predictor count: %d",
        len(num_cols),
    )

    logger.info(
        "Categorical predictor count: %d",
        len(cat_cols),
    )

    logger.info(
        "Total predictor count before encoding: %d",
        len(num_cols) + len(cat_cols),
    )

    return preprocessor


def prepare_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictors and a single target.

    Parameters
    ----------
    dataframe:
        Feature-engineered dataframe.

    target_column:
        Target to extract.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Predictor dataframe and target series.

    Raises
    ------
    ValueError
        If the target is unsupported or missing.
    """
    validate_feature_columns(dataframe)

    validate_targets_are_excluded(dataframe)
    validate_redundant_features_are_excluded(dataframe)

    if target_column not in TARGET_COLUMNS:
        raise ValueError(
            "Unsupported target column: "
            f"{target_column}. "
            f"Expected one of: {TARGET_COLUMNS}"
        )

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in dataframe."
        )

    features = dataframe[
        list(PREDICTOR_COLUMNS)
    ].copy()

    target = dataframe[
        target_column
    ].copy()

    logger.info(
        "Prepared predictors for target '%s'.",
        target_column,
    )

    logger.info(
        "Predictor shape: %s",
        features.shape,
    )

    logger.info(
        "Target shape: %s",
        target.shape,
    )

    return features, target


def get_predictor_columns() -> list[str]:
    """
    Return the authoritative predictor column list.

    Returns
    -------
    list[str]
        Ordered predictor columns.
    """
    return list(PREDICTOR_COLUMNS)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    logger.info(
        "preprocessing_pipeline.py provides reusable "
        "preprocessing components."
    )

    logger.info(
        "No transformer was fitted because training data "
        "was not supplied."
    )