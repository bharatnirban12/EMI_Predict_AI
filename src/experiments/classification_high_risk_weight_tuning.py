"""
High_Risk class-weight tuning experiment for LightGBM classification.

This experiment keeps the final tuned LightGBM configuration fixed and
changes only the relative class weight assigned to the High_Risk class.

Purpose
-------
Determine whether increasing the importance of High_Risk improves recall
and F1 without causing an unacceptable degradation in overall model
performance.

This experiment does NOT overwrite production model artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification_train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification_validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification_test.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
    / "classification_high_risk_weight_tuning_results.csv"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_COLUMN = "emi_eligibility"

RANDOM_STATE = 42

CLASS_LABELS = [
    "Eligible",
    "High_Risk",
    "Not_Eligible",
]

LABEL_MAPPING = {
    "Eligible": 0,
    "High_Risk": 1,
    "Not_Eligible": 2,
}

HIGH_RISK_WEIGHTS = [
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]


# Final tuned LightGBM configuration
LIGHTGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.08,
    "num_leaves": 63,
    "max_depth": 10,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multiclass",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------


def apply_feature_engineering(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the project regression/classification feature engineering.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe.
    """

    df = dataframe.copy()

    logger.info("Feature engineering completed successfully.")
    logger.info("Original feature count: %d", len(df.columns))

    # -------------------------------------------------------------
    # Required source columns
    # -------------------------------------------------------------

    required_columns = [
        "school_fees",
        "college_fees",
        "monthly_rent",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
        "monthly_salary",
        "current_emi_amount",
        "requested_amount",
        "requested_tenure",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns for feature engineering: "
            f"{missing_columns}"
        )

    # -------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------

    numeric_columns = [
        "school_fees",
        "college_fees",
        "monthly_rent",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
        "monthly_salary",
        "current_emi_amount",
        "requested_amount",
        "requested_tenure",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -------------------------------------------------------------
    # Aggregate expenses
    # -------------------------------------------------------------

    df["total_education_expenses"] = (
        df["school_fees"].fillna(0)
        + df["college_fees"].fillna(0)
    )

    df["total_monthly_living_expenses"] = (
        df["monthly_rent"].fillna(0)
        + df["travel_expenses"].fillna(0)
        + df["groceries_utilities"].fillna(0)
        + df["other_monthly_expenses"].fillna(0)
    )

    df["total_monthly_expenses"] = (
        df["total_monthly_living_expenses"]
        + df["total_education_expenses"]
        + df["current_emi_amount"].fillna(0)
    )

    # -------------------------------------------------------------
    # Disposable income
    # -------------------------------------------------------------

    df["disposable_income"] = (
        df["monthly_salary"]
        - df["total_monthly_expenses"]
    )

    # -------------------------------------------------------------
    # Ratios
    # -------------------------------------------------------------

    salary = df["monthly_salary"].replace(0, np.nan)

    df["expense_to_income_ratio"] = (
        df["total_monthly_expenses"] / salary
    )

    df["emi_to_income_ratio"] = (
        df["current_emi_amount"] / salary
    )

    df["requested_amount_to_income_ratio"] = (
        df["requested_amount"] / salary
    )

    # -------------------------------------------------------------
    # Requested amount per month
    # -------------------------------------------------------------

    tenure = df["requested_tenure"].replace(0, np.nan)

    df["requested_amount_per_month"] = (
        df["requested_amount"] / tenure
    )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    generated_features = [
        "total_education_expenses",
        "total_monthly_living_expenses",
        "total_monthly_expenses",
        "disposable_income",
        "expense_to_income_ratio",
        "emi_to_income_ratio",
        "requested_amount_to_income_ratio",
        "requested_amount_per_month",
    ]

    logger.info(
        "Final feature count: %d",
        len(df.columns),
    )

    logger.info(
        "Generated features: %s",
        generated_features,
    )

    return df


# ---------------------------------------------------------------------
# Predictor preparation
# ---------------------------------------------------------------------


def prepare_predictors(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare predictors and target.

    Parameters
    ----------
    dataframe:
        Feature-engineered dataframe.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Predictor dataframe and target series.
    """

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    df = dataframe.copy()

    target = df[TARGET_COLUMN].copy()

    predictors = df.drop(
        columns=[TARGET_COLUMN],
        errors="ignore",
    )

    # Remove target-related columns that must not be predictors.
    forbidden_columns = [
        "max_monthly_emi",
    ]

    predictors = predictors.drop(
        columns=[
            column
            for column in forbidden_columns
            if column in predictors.columns
        ],
        errors="ignore",
    )

    logger.info(
        "Prepared predictors for target '%s'.",
        TARGET_COLUMN,
    )

    logger.info(
        "Predictor shape: %s",
        predictors.shape,
    )

    logger.info(
        "Target shape: %s",
        target.shape,
    )

    return predictors, target


# ---------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------


def build_preprocessor(
    predictors: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Numerical columns are median-imputed and standardized.

    Categorical columns are most-frequent imputed and one-hot encoded.
    """

    numerical_columns = predictors.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = predictors.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    logger.info(
        "Categorical predictor count: %d",
        len(categorical_columns),
    )

    logger.info(
        "Numerical predictor count: %d",
        len(numerical_columns),
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ---------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------


def encode_target(
    target: pd.Series,
) -> np.ndarray:
    """
    Encode target labels using the project label mapping.
    """

    unknown_labels = set(target.dropna().unique()) - set(
        LABEL_MAPPING.keys()
    )

    if unknown_labels:
        raise ValueError(
            "Unknown target labels detected: "
            f"{sorted(unknown_labels)}"
        )

    encoded = target.map(LABEL_MAPPING)

    if encoded.isna().any():
        raise ValueError(
            "Target encoding produced missing values."
        )

    return encoded.astype(int).to_numpy()


# ---------------------------------------------------------------------
# Class weight construction
# ---------------------------------------------------------------------


def build_class_weights(
    high_risk_weight: float,
) -> Dict[int, float]:
    """
    Build class weights while changing only High_Risk weight.

    Parameters
    ----------
    high_risk_weight:
        Relative weight assigned to High_Risk.

    Returns
    -------
    dict[int, float]
        Class-weight mapping.
    """

    if high_risk_weight <= 0:
        raise ValueError(
            "High_Risk weight must be greater than zero."
        )

    return {
        LABEL_MAPPING["Eligible"]: 1.0,
        LABEL_MAPPING["High_Risk"]: high_risk_weight,
        LABEL_MAPPING["Not_Eligible"]: 1.0,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_model(
    model: LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    """

    predictions = model.predict(X)

    predictions = np.asarray(predictions).astype(int)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    weighted_precision = precision_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    high_risk_precision = precision_score(
        y,
        predictions,
        labels=[LABEL_MAPPING["High_Risk"]],
        average="macro",
        zero_division=0,
    )

    high_risk_recall = recall_score(
        y,
        predictions,
        labels=[LABEL_MAPPING["High_Risk"]],
        average="macro",
        zero_division=0,
    )

    high_risk_f1 = f1_score(
        y,
        predictions,
        labels=[LABEL_MAPPING["High_Risk"]],
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "high_risk_precision": float(
            high_risk_precision
        ),
        "high_risk_recall": float(
            high_risk_recall
        ),
        "high_risk_f1": float(
            high_risk_f1
        ),
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------


def main() -> None:
    """
    Run the High_Risk class-weight experiment.
    """

    logger.info(
        "Starting High_Risk class-weight tuning experiment."
    )

    # -------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    logger.info(
        "Loaded training dataset: %s",
        train_df.shape,
    )

    logger.info(
        "Loaded validation dataset: %s",
        validation_df.shape,
    )

    logger.info(
        "Loaded test dataset: %s",
        test_df.shape,
    )

    # -------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------

    train_df = apply_feature_engineering(train_df)
    validation_df = apply_feature_engineering(
        validation_df
    )
    test_df = apply_feature_engineering(test_df)

    # -------------------------------------------------------------
    # Prepare predictors
    # -------------------------------------------------------------

    X_train_raw, y_train_raw = prepare_predictors(
        train_df
    )

    X_validation_raw, y_validation_raw = (
        prepare_predictors(validation_df)
    )

    X_test_raw, y_test_raw = prepare_predictors(
        test_df
    )

    # -------------------------------------------------------------
    # Encode targets
    # -------------------------------------------------------------

    y_train = encode_target(y_train_raw)
    y_validation = encode_target(y_validation_raw)
    y_test = encode_target(y_test_raw)

    logger.info(
        "Target mapping: %s",
        LABEL_MAPPING,
    )

    # -------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------

    preprocessor = build_preprocessor(
        X_train_raw
    )

    logger.info(
        "Fitting preprocessing pipeline on training data only."
    )

    X_train = preprocessor.fit_transform(
        X_train_raw
    )

    logger.info(
        "Transforming validation data."
    )

    X_validation = preprocessor.transform(
        X_validation_raw
    )

    logger.info(
        "Transforming test data."
    )

    X_test = preprocessor.transform(
        X_test_raw
    )

    logger.info(
        "Transformed training shape: %s",
        X_train.shape,
    )

    logger.info(
        "Transformed validation shape: %s",
        X_validation.shape,
    )

    logger.info(
        "Transformed test shape: %s",
        X_test.shape,
    )

    # -------------------------------------------------------------
    # Experiment
    # -------------------------------------------------------------

    results: List[Dict[str, float]] = []

    for high_risk_weight in HIGH_RISK_WEIGHTS:

        logger.info(
            "===================================================="
        )

        logger.info(
            "Testing High_Risk class weight: %.2f",
            high_risk_weight,
        )

        class_weights = build_class_weights(
            high_risk_weight
        )

        logger.info(
            "Class weights: %s",
            class_weights,
        )

        sample_weights = np.array(
            [
                class_weights[label]
                for label in y_train
            ],
            dtype=float,
        )

        model = LGBMClassifier(
            **LIGHTGBM_PARAMS,
            class_weight=None,
        )

        logger.info(
            "Training LightGBM with High_Risk weight %.2f.",
            high_risk_weight,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

        validation_metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        result = {
            "high_risk_weight": high_risk_weight,

            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
            "validation_weighted_precision": (
                validation_metrics["weighted_precision"]
            ),
            "validation_weighted_recall": (
                validation_metrics["weighted_recall"]
            ),
            "validation_weighted_f1": (
                validation_metrics["weighted_f1"]
            ),
            "validation_macro_f1": (
                validation_metrics["macro_f1"]
            ),
            "validation_high_risk_precision": (
                validation_metrics["high_risk_precision"]
            ),
            "validation_high_risk_recall": (
                validation_metrics["high_risk_recall"]
            ),
            "validation_high_risk_f1": (
                validation_metrics["high_risk_f1"]
            ),

            "test_accuracy": (
                test_metrics["accuracy"]
            ),
            "test_weighted_precision": (
                test_metrics["weighted_precision"]
            ),
            "test_weighted_recall": (
                test_metrics["weighted_recall"]
            ),
            "test_weighted_f1": (
                test_metrics["weighted_f1"]
            ),
            "test_macro_f1": (
                test_metrics["macro_f1"]
            ),
            "test_high_risk_precision": (
                test_metrics["high_risk_precision"]
            ),
            "test_high_risk_recall": (
                test_metrics["high_risk_recall"]
            ),
            "test_high_risk_f1": (
                test_metrics["high_risk_f1"]
            ),
        }

        results.append(result)

        logger.info(
            "Validation High_Risk Precision: %.4f",
            validation_metrics[
                "high_risk_precision"
            ],
        )

        logger.info(
            "Validation High_Risk Recall: %.4f",
            validation_metrics[
                "high_risk_recall"
            ],
        )

        logger.info(
            "Validation High_Risk F1: %.4f",
            validation_metrics[
                "high_risk_f1"
            ],
        )

        logger.info(
            "Test High_Risk Precision: %.4f",
            test_metrics[
                "high_risk_precision"
            ],
        )

        logger.info(
            "Test High_Risk Recall: %.4f",
            test_metrics[
                "high_risk_recall"
            ],
        )

        logger.info(
            "Test High_Risk F1: %.4f",
            test_metrics[
                "high_risk_f1"
            ],
        )

    # -------------------------------------------------------------
    # Results dataframe
    # -------------------------------------------------------------

    results_df = pd.DataFrame(results)

    # Rank using validation High_Risk F1.
    results_df = results_df.sort_values(
        by=[
            "validation_high_risk_f1",
            "validation_macro_f1",
        ],
        ascending=False,
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "rank",
        np.arange(1, len(results_df) + 1),
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------

    best = results_df.iloc[0]

    print()
    print("=" * 90)
    print("LIGHTGBM HIGH_RISK CLASS-WEIGHT TUNING")
    print("=" * 90)

    print(
        f"Configurations evaluated: "
        f"{len(results_df)}"
    )

    print()
    print("Best Configuration:")
    print("-" * 90)
    print(
        f"High_Risk weight: "
        f"{best['high_risk_weight']:.2f}"
    )

    print()
    print("Validation Metrics:")
    print("-" * 90)
    print(
        f"Accuracy:              "
        f"{best['validation_accuracy']:.4f}"
    )
    print(
        f"Weighted F1:           "
        f"{best['validation_weighted_f1']:.4f}"
    )
    print(
        f"Macro F1:              "
        f"{best['validation_macro_f1']:.4f}"
    )
    print(
        f"High_Risk Precision:   "
        f"{best['validation_high_risk_precision']:.4f}"
    )
    print(
        f"High_Risk Recall:      "
        f"{best['validation_high_risk_recall']:.4f}"
    )
    print(
        f"High_Risk F1:          "
        f"{best['validation_high_risk_f1']:.4f}"
    )

    print()
    print("Test Metrics:")
    print("-" * 90)
    print(
        f"Accuracy:              "
        f"{best['test_accuracy']:.4f}"
    )
    print(
        f"Weighted F1:           "
        f"{best['test_weighted_f1']:.4f}"
    )
    print(
        f"Macro F1:              "
        f"{best['test_macro_f1']:.4f}"
    )
    print(
        f"High_Risk Precision:   "
        f"{best['test_high_risk_precision']:.4f}"
    )
    print(
        f"High_Risk Recall:      "
        f"{best['test_high_risk_recall']:.4f}"
    )
    print(
        f"High_Risk F1:          "
        f"{best['test_high_risk_f1']:.4f}"
    )

    print()
    print("All Configurations:")
    print("-" * 90)

    display_columns = [
        "rank",
        "high_risk_weight",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_high_risk_precision",
        "validation_high_risk_recall",
        "validation_high_risk_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_high_risk_precision",
        "test_high_risk_recall",
        "test_high_risk_f1",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("Results saved:")
    print(OUTPUT_PATH)

    logger.info(
        "High_Risk class-weight tuning completed."
    )


if __name__ == "__main__":
    main()