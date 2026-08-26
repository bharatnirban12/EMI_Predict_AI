"""
Empirical target leakage audit for the EMI prediction dataset.

This module performs relationship and proxy analysis between predictor
variables and the two supplied target variables:

    - emi_eligibility
    - max_monthly_emi

Important:
    This module does not train machine-learning models.
    It does not modify the raw dataset.
    It does not impute missing values.
    It does not remove outliers.

The purpose is to identify unusually strong or potentially deterministic
relationships that require further investigation before model training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


TARGET_CLASSIFICATION = "emi_eligibility"
TARGET_REGRESSION = "max_monthly_emi"


RAW_NUMERIC_COLUMNS = [
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


RAW_CATEGORICAL_COLUMNS = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
]


ENGINEERED_NUMERIC_COLUMNS = [
    "total_education_expenses",
    "total_monthly_living_expenses",
    "total_monthly_expenses",
    "disposable_income",
    "expense_to_income_ratio",
    "emi_to_income_ratio",
    "requested_amount_to_income_ratio",
    "requested_amount_per_month",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run empirical target leakage audit."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the raw EMI dataset.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory for leakage-audit tables.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load and validate the raw dataset.

    Parameters
    ----------
    data_path:
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    LOGGER.info("Loading raw dataset: %s", data_path)

    dataframe = pd.read_csv(data_path)

    required_columns = (
        RAW_NUMERIC_COLUMNS
        + RAW_CATEGORICAL_COLUMNS
        + [
            TARGET_CLASSIFICATION,
            TARGET_REGRESSION,
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )

    LOGGER.info(
        "Dataset loaded successfully: %s",
        dataframe.shape,
    )

    return dataframe


def convert_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create numeric analysis columns without imputing values.

    Invalid numeric strings become NaN.

    The original dataframe is not modified.
    """
    result = dataframe.copy()

    numeric_columns = (
        RAW_NUMERIC_COLUMNS
        + [TARGET_REGRESSION]
    )

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def add_engineered_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the project's documented engineered financial features.

    No missing-value imputation is performed.
    """
    result = dataframe.copy()

    result["total_education_expenses"] = (
        result["school_fees"]
        + result["college_fees"]
    )

    result["total_monthly_living_expenses"] = (
        result["travel_expenses"]
        + result["groceries_utilities"]
        + result["other_monthly_expenses"]
    )

    result["total_monthly_expenses"] = (
        result["total_education_expenses"]
        + result["total_monthly_living_expenses"]
        + result["current_emi_amount"]
    )

    result["disposable_income"] = (
        result["monthly_salary"]
        - result["total_monthly_expenses"]
    )

    salary = result["monthly_salary"].replace(
        0,
        np.nan,
    )

    result["expense_to_income_ratio"] = (
        result["total_monthly_expenses"]
        / salary
    )

    result["emi_to_income_ratio"] = (
        result["current_emi_amount"]
        / salary
    )

    result["requested_amount_to_income_ratio"] = (
        result["requested_amount"]
        / salary
    )

    tenure = result["requested_tenure"].replace(
        0,
        np.nan,
    )

    result["requested_amount_per_month"] = (
        result["requested_amount"]
        / tenure
    )

    return result


def calculate_univariate_regression_r2(
    feature: pd.Series,
    target: pd.Series,
) -> float:
    """
    Calculate R² from a simple univariate linear relationship.

    This is analytical EDA, not model training.
    """
    valid = pd.concat(
        [feature, target],
        axis=1,
    ).dropna()

    if len(valid) < 2:
        return float("nan")

    x = valid.iloc[:, 0].to_numpy(dtype=float)
    y = valid.iloc[:, 1].to_numpy(dtype=float)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    denominator = np.sum(
        (x - x_mean) ** 2
    )

    if denominator == 0:
        return float("nan")

    slope = np.sum(
        (x - x_mean) * (y - y_mean)
    ) / denominator

    intercept = y_mean - slope * x_mean

    predictions = (
        slope * x + intercept
    )

    residual_sum_of_squares = np.sum(
        (y - predictions) ** 2
    )

    total_sum_of_squares = np.sum(
        (y - y_mean) ** 2
    )

    if total_sum_of_squares == 0:
        return float("nan")

    return float(
        1.0
        - (
            residual_sum_of_squares
            / total_sum_of_squares
        )
    )


def regression_numeric_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze numeric predictors against max_monthly_emi.
    """
    features = (
        RAW_NUMERIC_COLUMNS
        + ENGINEERED_NUMERIC_COLUMNS
    )

    target = dataframe[TARGET_REGRESSION]

    rows: List[Dict[str, object]] = []

    for feature_name in features:
        feature = dataframe[feature_name]

        valid = pd.concat(
            [feature, target],
            axis=1,
        ).dropna()

        if len(valid) < 2:
            rows.append(
                {
                    "feature": feature_name,
                    "valid_records": len(valid),
                    "pearson_correlation": np.nan,
                    "spearman_correlation": np.nan,
                    "univariate_linear_r2": np.nan,
                }
            )
            continue

        pearson = valid.iloc[:, 0].corr(
            valid.iloc[:, 1],
            method="pearson",
        )

        spearman = valid.iloc[:, 0].corr(
            valid.iloc[:, 1],
            method="spearman",
        )

        r2 = calculate_univariate_regression_r2(
            valid.iloc[:, 0],
            valid.iloc[:, 1],
        )

        rows.append(
            {
                "feature": feature_name,
                "valid_records": len(valid),
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "univariate_linear_r2": r2,
            }
        )

    result = pd.DataFrame(rows)

    result["absolute_pearson"] = (
        result["pearson_correlation"].abs()
    )

    result["absolute_spearman"] = (
        result["spearman_correlation"].abs()
    )

    result = result.sort_values(
        "absolute_spearman",
        ascending=False,
    )

    return result


def regression_binned_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze target behavior across predictor quantiles.
    """
    features = (
        RAW_NUMERIC_COLUMNS
        + ENGINEERED_NUMERIC_COLUMNS
    )

    rows: List[Dict[str, object]] = []

    for feature_name in features:
        working = dataframe[
            [feature_name, TARGET_REGRESSION]
        ].dropna()

        if len(working) < 20:
            continue

        if working[feature_name].nunique() < 4:
            continue

        try:
            working["feature_bin"] = pd.qcut(
                working[feature_name],
                q=10,
                duplicates="drop",
            )
        except ValueError:
            continue

        grouped = (
            working
            .groupby(
                "feature_bin",
                observed=True,
            )[TARGET_REGRESSION]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "feature": feature_name,
                    "feature_bin": str(
                        row["feature_bin"]
                    ),
                    "count": int(row["count"]),
                    "target_mean": float(row["mean"]),
                    "target_median": float(
                        row["median"]
                    ),
                    "target_std": float(
                        row["std"]
                    )
                    if pd.notna(row["std"])
                    else np.nan,
                    "target_minimum": float(
                        row["minimum"]
                    ),
                    "target_maximum": float(
                        row["maximum"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def classification_numeric_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze numeric predictors by classification target.
    """
    features = (
        RAW_NUMERIC_COLUMNS
        + ENGINEERED_NUMERIC_COLUMNS
    )

    rows: List[Dict[str, object]] = []

    for feature_name in features:
        grouped = (
            dataframe
            .groupby(TARGET_CLASSIFICATION)[
                feature_name
            ]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "feature": feature_name,
                    "target_class": row[
                        TARGET_CLASSIFICATION
                    ],
                    "count": int(row["count"]),
                    "mean": float(row["mean"])
                    if pd.notna(row["mean"])
                    else np.nan,
                    "median": float(row["median"])
                    if pd.notna(row["median"])
                    else np.nan,
                    "std": float(row["std"])
                    if pd.notna(row["std"])
                    else np.nan,
                    "minimum": float(row["minimum"])
                    if pd.notna(row["minimum"])
                    else np.nan,
                    "maximum": float(row["maximum"])
                    if pd.notna(row["maximum"])
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def classification_categorical_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze categorical predictors against classification target.
    """
    rows: List[Dict[str, object]] = []

    for feature_name in RAW_CATEGORICAL_COLUMNS:
        counts = pd.crosstab(
            dataframe[feature_name],
            dataframe[TARGET_CLASSIFICATION],
        )

        proportions = pd.crosstab(
            dataframe[feature_name],
            dataframe[TARGET_CLASSIFICATION],
            normalize="index",
        )

        for category in counts.index:
            total = counts.loc[category].sum()

            for target_class in counts.columns:
                rows.append(
                    {
                        "feature": feature_name,
                        "category": category,
                        "target_class": target_class,
                        "count": int(
                            counts.loc[
                                category,
                                target_class,
                            ]
                        ),
                        "category_total": int(total),
                        "target_percentage": float(
                            proportions.loc[
                                category,
                                target_class,
                            ]
                            * 100.0
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_summary(
    regression_correlations: pd.DataFrame,
    classification_numeric: pd.DataFrame,
    classification_categorical: pd.DataFrame,
) -> Dict[str, object]:
    """
    Build a machine-readable audit summary.
    """
    top_regression = (
        regression_correlations
        .head(10)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    high_risk_summary = (
        classification_numeric[
            classification_numeric["target_class"]
            == "High_Risk"
        ]
        .head(0)
        .to_dict(orient="records")
    )

    return {
        "purpose": (
            "Empirical investigation of predictor-target "
            "relationships and potential proxy relationships."
        ),
        "target_generation_formula_available": False,
        "interpretation_warning": (
            "Strong statistical relationships do not by "
            "themselves prove target leakage."
        ),
        "top_regression_relationships": top_regression,
        "high_risk_numeric_records": high_risk_summary,
        "classification_categorical_rows": int(
            len(classification_categorical)
        ),
    }


def save_outputs(
    output_directory: Path,
    regression_correlations: pd.DataFrame,
    regression_binned: pd.DataFrame,
    classification_numeric: pd.DataFrame,
    classification_categorical: pd.DataFrame,
    summary: Dict[str, object],
) -> None:
    """Save all audit outputs."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    regression_correlations.to_csv(
        output_directory
        / "leakage_numeric_correlations.csv",
        index=False,
    )

    regression_binned.to_csv(
        output_directory
        / "leakage_regression_binned_summary.csv",
        index=False,
    )

    classification_numeric.to_csv(
        output_directory
        / "leakage_classification_numeric_summary.csv",
        index=False,
    )

    classification_categorical.to_csv(
        output_directory
        / "leakage_classification_categorical_summary.csv",
        index=False,
    )

    with open(
        output_directory
        / "leakage_audit_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    LOGGER.info(
        "Saved target leakage audit outputs to: %s",
        output_directory,
    )


def main() -> None:
    """Run the target leakage audit."""
    configure_logging()

    args = parse_args()

    dataframe = load_dataset(
        args.data_path
    )

    dataframe = convert_numeric_columns(
        dataframe
    )

    dataframe = add_engineered_features(
        dataframe
    )

    LOGGER.info(
        "Engineered features created: %s",
        ENGINEERED_NUMERIC_COLUMNS,
    )

    regression_correlations = (
        regression_numeric_analysis(
            dataframe
        )
    )

    regression_binned = (
        regression_binned_analysis(
            dataframe
        )
    )

    classification_numeric = (
        classification_numeric_analysis(
            dataframe
        )
    )

    classification_categorical = (
        classification_categorical_analysis(
            dataframe
        )
    )

    summary = build_summary(
        regression_correlations,
        classification_numeric,
        classification_categorical,
    )

    save_outputs(
        output_directory=args.output_directory,
        regression_correlations=regression_correlations,
        regression_binned=regression_binned,
        classification_numeric=classification_numeric,
        classification_categorical=classification_categorical,
        summary=summary,
    )

    LOGGER.info(
        "Target leakage audit completed successfully."
    )


if __name__ == "__main__":
    main()