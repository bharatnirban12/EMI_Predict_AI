"""
Targeted error analysis for the final LightGBM classification model.

This module analyzes actual High_Risk test observations and separates them
into:

1. Correctly predicted High_Risk
2. High_Risk predicted as Eligible
3. High_Risk predicted as Not_Eligible

The analysis uses the same feature-engineering implementation as the model
pipeline and produces reproducible CSV reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.features.feature_engineering import engineer_features


LOGGER = logging.getLogger(__name__)

TEST_DATA_PATH = Path(
    "data/processed/classification_test.csv"
)

ERROR_ANALYSIS_PATH = Path(
    "artifacts/classification/evaluation/"
    "lightgbm_tuned_error_analysis.csv"
)

OUTPUT_DIRECTORY = Path(
    "reports/modeling/classification_error_analysis"
)

TARGET_COLUMN = "emi_eligibility"
ACTUAL_COLUMN = "actual_emi_eligibility"
PREDICTED_COLUMN = "predicted_emi_eligibility"


NUMERICAL_COLUMNS: List[str] = [
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
    "total_education_expenses",
    "total_monthly_living_expenses",
    "total_monthly_expenses",
    "disposable_income",
    "expense_to_income_ratio",
    "emi_to_income_ratio",
    "requested_amount_to_income_ratio",
    "requested_amount_per_month",
]

CATEGORICAL_COLUMNS: List[str] = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
]


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_test_data() -> pd.DataFrame:
    """Load the classification test dataset."""
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    LOGGER.info("Loading test dataset: %s", TEST_DATA_PATH)

    dataframe = pd.read_csv(
        TEST_DATA_PATH,
        low_memory=False,
    )

    LOGGER.info(
        "Loaded test dataset: %s",
        dataframe.shape,
    )

    return dataframe


def load_error_analysis() -> pd.DataFrame:
    """Load the existing model error-analysis artifact."""
    if not ERROR_ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            "Error-analysis artifact not found: "
            f"{ERROR_ANALYSIS_PATH}"
        )

    LOGGER.info(
        "Loading error-analysis artifact: %s",
        ERROR_ANALYSIS_PATH,
    )

    dataframe = pd.read_csv(
        ERROR_ANALYSIS_PATH,
        low_memory=False,
    )

    LOGGER.info(
        "Loaded error-analysis artifact: %s",
        dataframe.shape,
    )

    return dataframe


def validate_error_artifact(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the existing error-analysis artifact."""
    required_columns = {
        ACTUAL_COLUMN,
        PREDICTED_COLUMN,
        "prediction_correct",
        "prediction_error",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required error-analysis columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_columns = dataframe.columns[
        dataframe.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise ValueError(
            "Duplicate columns found: "
            f"{duplicate_columns}"
        )

    high_risk_errors = dataframe[
        dataframe[ACTUAL_COLUMN] == "High_Risk"
    ]

    expected_total_errors = 741

    high_risk_error_count = len(
        high_risk_errors
    )

    if high_risk_error_count != expected_total_errors:
        raise ValueError(
            "Unexpected High_Risk error count. "
            f"Expected {expected_total_errors}, "
            f"found {high_risk_error_count}."
        )

    high_risk_to_eligible = len(
        high_risk_errors[
            high_risk_errors[PREDICTED_COLUMN]
            == "Eligible"
        ]
    )

    high_risk_to_not_eligible = len(
        high_risk_errors[
            high_risk_errors[PREDICTED_COLUMN]
            == "Not_Eligible"
        ]
    )

    if high_risk_to_eligible != 337:
        raise ValueError(
            "Unexpected High_Risk → Eligible count. "
            f"Expected 337, found {high_risk_to_eligible}."
        )

    if high_risk_to_not_eligible != 404:
        raise ValueError(
            "Unexpected High_Risk → Not_Eligible count. "
            f"Expected 404, found {high_risk_to_not_eligible}."
        )

    if (
        high_risk_to_eligible
        + high_risk_to_not_eligible
        != expected_total_errors
    ):
        raise ValueError(
            "High_Risk error groups do not sum to "
            "the expected total."
        )

    LOGGER.info(
        "Error artifact validation passed."
    )


def prepare_high_risk_population(
    test_data: pd.DataFrame,
    error_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the complete High_Risk population.

    The original test dataset provides the 2,546 actual High_Risk
    observations. The error artifact provides prediction information
    for the 1,243 misclassified observations.

    Correct High_Risk observations are recovered by excluding all
    error records from the actual High_Risk test population.
    """
    if TARGET_COLUMN not in test_data.columns:
        raise ValueError(
            f"Target column not found: {TARGET_COLUMN}"
        )

    actual_high_risk = test_data[
        test_data[TARGET_COLUMN] == "High_Risk"
    ].copy()

    if len(actual_high_risk) != 2546:
        raise ValueError(
            "Unexpected actual High_Risk count. "
            f"Expected 2546, found {len(actual_high_risk)}."
        )

    error_high_risk = error_data[
        error_data[ACTUAL_COLUMN] == "High_Risk"
    ].copy()

    error_high_risk = error_high_risk[
        [
            ACTUAL_COLUMN,
            PREDICTED_COLUMN,
        ]
        + [
            column
            for column in error_high_risk.columns
            if column in test_data.columns
        ]
    ]

    test_columns = [
        column
        for column in test_data.columns
        if column in error_high_risk.columns
    ]

    if not test_columns:
        raise ValueError(
            "No common predictor columns found between "
            "test data and error artifact."
        )

    # The error artifact contains the original test rows.
    # Use stable row-level matching across all original columns.
    merge_columns = [
        column
        for column in test_data.columns
        if column in error_high_risk.columns
    ]

    error_keys = error_high_risk[merge_columns].copy()

    error_keys = error_keys.drop_duplicates()

    error_keys["_is_error"] = True

    merged = actual_high_risk.merge(
        error_keys,
        on=merge_columns,
        how="left",
        indicator=False,
        suffixes=("", "_error"),
    )

    merged["_is_error"] = (
        merged["_is_error"]
        .fillna(False)
        .astype(bool)
    )

    correct_high_risk = merged[
        ~merged["_is_error"]
    ].copy()

    if len(correct_high_risk) != 1805:
        raise ValueError(
            "Unable to recover expected number of "
            "correct High_Risk observations. "
            f"Expected 1805, found {len(correct_high_risk)}."
        )

    correct_high_risk[
        PREDICTED_COLUMN
    ] = "High_Risk"

    correct_high_risk[
        "error_group"
    ] = "Correct_High_Risk"

    error_high_risk = error_data[
        error_data[ACTUAL_COLUMN] == "High_Risk"
    ].copy()

    error_high_risk[
        "error_group"
    ] = error_high_risk[
        PREDICTED_COLUMN
    ].map(
        {
            "Eligible": "High_Risk_to_Eligible",
            "Not_Eligible": "High_Risk_to_Not_Eligible",
        }
    )

    error_high_risk = error_high_risk[
        error_high_risk["error_group"].notna()
    ].copy()

    common_columns = [
        column
        for column in test_data.columns
        if column in error_high_risk.columns
    ]

    error_population = error_high_risk[
        common_columns
        + [
            PREDICTED_COLUMN,
            "error_group",
        ]
    ].copy()

    correct_population = correct_high_risk[
        common_columns
        + [
            PREDICTED_COLUMN,
            "error_group",
        ]
    ].copy()

    population = pd.concat(
        [
            correct_population,
            error_population,
        ],
        ignore_index=True,
    )

    if len(population) != 2546:
        raise ValueError(
            "Final High_Risk population has unexpected "
            f"size: {len(population)}."
        )

    return population


def add_engineered_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the project's existing feature engineering."""
    LOGGER.info(
        "Applying project feature engineering."
    )

    engineered = engineer_features(
        dataframe.copy()
    )

    LOGGER.info(
        "Engineered feature shape: %s",
        engineered.shape,
    )

    return engineered


def numerical_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Generate numerical summaries by error group."""
    available = [
        column
        for column in NUMERICAL_COLUMNS
        if column in dataframe.columns
    ]

    records = []

    for feature in available:
        for group, group_data in dataframe.groupby(
            "error_group",
            dropna=False,
        ):
            values = pd.to_numeric(
                group_data[feature],
                errors="coerce",
            )

            records.append(
                {
                    "feature": feature,
                    "error_group": group,
                    "count": int(values.notna().sum()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "min": values.min(),
                    "p25": values.quantile(0.25),
                    "p50": values.quantile(0.50),
                    "p75": values.quantile(0.75),
                    "p90": values.quantile(0.90),
                    "p95": values.quantile(0.95),
                    "max": values.max(),
                }
            )

    return pd.DataFrame(records)


def categorical_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Generate categorical distributions by error group."""
    available = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in dataframe.columns
    ]

    records = []

    for feature in available:
        counts = (
            dataframe.groupby(
                ["error_group", feature],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )

        totals = (
            dataframe.groupby(
                "error_group",
                dropna=False,
            )
            .size()
            .rename("group_total")
            .reset_index()
        )

        counts = counts.merge(
            totals,
            on="error_group",
            how="left",
        )

        counts["percentage"] = (
            counts["count"]
            / counts["group_total"]
            * 100.0
        )

        counts["feature"] = feature

        records.append(
            counts[
                [
                    "feature",
                    "error_group",
                    feature,
                    "count",
                    "group_total",
                    "percentage",
                ]
            ].rename(
                columns={feature: "category"}
            )
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "feature",
                "error_group",
                "category",
                "count",
                "group_total",
                "percentage",
            ]
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


def create_population_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create high-level population counts."""
    summary = (
        dataframe.groupby(
            "error_group",
            dropna=False,
        )
        .size()
        .reset_index(name="records")
    )

    total = summary["records"].sum()

    summary["percentage_of_high_risk"] = (
        summary["records"] / total * 100.0
    )

    return summary


def create_error_patterns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact feature-pattern report.

    For numerical variables, compare group medians and means.
    For categorical variables, report the most frequent category
    in each group.
    """
    records = []

    for feature in NUMERICAL_COLUMNS:
        if feature not in dataframe.columns:
            continue

        group_stats = (
            dataframe.groupby(
                "error_group",
                dropna=False,
            )[feature]
            .agg(
                count="count",
                mean="mean",
                median="median",
            )
            .reset_index()
        )

        for _, row in group_stats.iterrows():
            records.append(
                {
                    "feature_type": "numerical",
                    "feature": feature,
                    "error_group": row[
                        "error_group"
                    ],
                    "count": row["count"],
                    "mean": row["mean"],
                    "median": row["median"],
                    "category": None,
                    "percentage": None,
                }
            )

    for feature in CATEGORICAL_COLUMNS:
        if feature not in dataframe.columns:
            continue

        counts = (
            dataframe.groupby(
                ["error_group", feature],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )

        totals = (
            dataframe.groupby(
                "error_group",
                dropna=False,
            )
            .size()
            .rename("group_total")
            .reset_index()
        )

        counts = counts.merge(
            totals,
            on="error_group",
            how="left",
        )

        counts["percentage"] = (
            counts["count"]
            / counts["group_total"]
            * 100.0
        )

        top_categories = (
            counts.sort_values(
                [
                    "error_group",
                    "count",
                ],
                ascending=[True, False],
            )
            .groupby(
                "error_group",
                as_index=False,
            )
            .head(3)
        )

        for _, row in top_categories.iterrows():
            records.append(
                {
                    "feature_type": "categorical",
                    "feature": feature,
                    "error_group": row[
                        "error_group"
                    ],
                    "count": row["count"],
                    "mean": None,
                    "median": None,
                    "category": row[feature],
                    "percentage": row["percentage"],
                }
            )

    return pd.DataFrame(records)


def save_outputs(
    population_summary: pd.DataFrame,
    numerical: pd.DataFrame,
    patterns: pd.DataFrame,
) -> Dict[str, Path]:
    """Save analysis outputs."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "population": (
            OUTPUT_DIRECTORY
            / "high_risk_error_summary.csv"
        ),
        "numerical": (
            OUTPUT_DIRECTORY
            / "high_risk_error_feature_summary.csv"
        ),
        "patterns": (
            OUTPUT_DIRECTORY
            / "high_risk_error_patterns.csv"
        ),
    }

    population_summary.to_csv(
        paths["population"],
        index=False,
    )

    numerical.to_csv(
        paths["numerical"],
        index=False,
    )

    patterns.to_csv(
        paths["patterns"],
        index=False,
    )

    for path in paths.values():
        LOGGER.info(
            "Saved output: %s",
            path,
        )

    return paths


def main() -> None:
    """Run targeted High_Risk error analysis."""
    configure_logging()

    LOGGER.info(
        "Starting High_Risk classification error analysis."
    )

    test_data = load_test_data()

    error_data = load_error_analysis()

    validate_error_artifact(error_data)

    population = prepare_high_risk_population(
        test_data,
        error_data,
    )

    population = add_engineered_features(
        population
    )

    population_summary = create_population_summary(
        population
    )

    numerical = numerical_summary(
        population
    )

    categorical = categorical_summary(
        population
    )

    patterns = create_error_patterns(
        population
    )

    # Combine numerical and categorical summaries
    # into the pattern report without changing either
    # underlying analysis.
    if not categorical.empty:
        categorical_patterns = categorical.copy()

        categorical_patterns[
            "feature_type"
        ] = "categorical"

        categorical_patterns[
            "mean"
        ] = None

        categorical_patterns[
            "median"
        ] = None

        categorical_patterns[
            "category"
        ] = categorical_patterns["category"]

        categorical_patterns = (
            categorical_patterns[
                [
                    "feature_type",
                    "feature",
                    "error_group",
                    "count",
                    "mean",
                    "median",
                    "category",
                    "percentage",
                ]
            ]
        )

        patterns = pd.concat(
            [
                patterns,
                categorical_patterns,
            ],
            ignore_index=True,
        )

    paths = save_outputs(
        population_summary,
        numerical,
        patterns,
    )

    print()
    print("=" * 70)
    print("HIGH_RISK CLASSIFICATION ERROR ANALYSIS")
    print("=" * 70)

    print()
    print(
        "High_Risk population:",
        len(population),
    )

    print()
    print(
        population_summary.to_string(
            index=False
        )
    )

    print()
    print("Validation:")
    print("-" * 70)

    counts = population[
        "error_group"
    ].value_counts()

    print(
        "Correct High_Risk:",
        counts.get(
            "Correct_High_Risk",
            0,
        ),
    )

    print(
        "High_Risk -> Eligible:",
        counts.get(
            "High_Risk_to_Eligible",
            0,
        ),
    )

    print(
        "High_Risk -> Not_Eligible:",
        counts.get(
            "High_Risk_to_Not_Eligible",
            0,
        ),
    )

    print()
    print("Outputs:")
    print("-" * 70)

    for path in paths.values():
        print(path)

    LOGGER.info(
        "High_Risk classification error analysis completed."
    )


if __name__ == "__main__":
    main()