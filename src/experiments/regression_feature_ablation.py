"""
Run XGBoost feature-ablation experiments for EMI regression.

The experiment compares the existing engineered-feature configuration
against controlled feature-removal variants.

Important:
    The train/validation/test split is fixed.
    The XGBoost configuration is fixed.
    Only the predictor feature set changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from src.features.feature_engineering import engineer_features
from src.preprocessing.preprocessing_pipeline import (
    create_preprocessing_pipeline,
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"

TRAIN_PATH = DATA_DIR / "regression_train.csv"
VALIDATION_PATH = DATA_DIR / "regression_validation.csv"
TEST_PATH = DATA_DIR / "regression_test.csv"

TARGET_COLUMN = "max_monthly_emi"

ENGINEERED_FEATURES = [
    "total_education_expenses",
    "total_monthly_living_expenses",
    "total_monthly_expenses",
    "disposable_income",
    "expense_to_income_ratio",
    "emi_to_income_ratio",
    "requested_amount_to_income_ratio",
    "requested_amount_per_month",
]

FINANCIAL_RATIO_FEATURES = [
    "disposable_income",
    "expense_to_income_ratio",
    "emi_to_income_ratio",
    "requested_amount_to_income_ratio",
    "requested_amount_per_month",
]

N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.05
SUBSAMPLE = 0.80
COLSAMPLE_BYTREE = 0.80
RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    """Load one regression split."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"Dataset is empty: {path}"
        )

    logger.info(
        "Loaded %s: %s",
        path.name,
        dataframe.shape,
    )

    return dataframe


def prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply feature engineering and separate target."""
    engineered = engineer_features(dataframe)

    features, target = prepare_features_and_target(
        engineered,
        TARGET_COLUMN,
    )

    if TARGET_COLUMN in features.columns:
        raise ValueError(
            f"Target leakage detected: {TARGET_COLUMN}"
        )

    return features, target


def remove_features(
    features: pd.DataFrame,
    features_to_remove: list[str],
) -> pd.DataFrame:
    """Return a copy with selected features removed."""
    missing = [
        column
        for column in features_to_remove
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            "Requested features were not found: "
            f"{missing}"
        )

    return features.drop(
        columns=features_to_remove
    ).copy()


def validate_feature_columns(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> None:
    """Ensure all splits have identical predictor columns."""
    train_columns = list(train_features.columns)
    validation_columns = list(validation_features.columns)
    test_columns = list(test_features.columns)

    if train_columns != validation_columns:
        raise ValueError(
            "Training and validation feature columns differ."
        )

    if train_columns != test_columns:
        raise ValueError(
            "Training and test feature columns differ."
        )


def evaluate(
    model: XGBRegressor,
    features,
    target: pd.Series,
) -> dict[str, float]:
    """Calculate regression metrics."""
    predictions = model.predict(features)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model predictions contain NaN or infinite values."
        )

    mae = mean_absolute_error(
        target,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            target,
            predictions,
        )
    )

    r2 = r2_score(
        target,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def train_variant(
    name: str,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Train and evaluate one feature-set variant."""
    logger.info(
        "Starting feature variant: %s",
        name,
    )

    validate_feature_columns(
        train_features,
        validation_features,
        test_features,
    )

    logger.info(
        "%s predictor count: %d",
        name,
        train_features.shape[1],
    )

    preprocessor = create_preprocessing_pipeline(
        train_features, allow_missing=True
    )

    X_train = preprocessor.fit_transform(
        train_features
    )

    X_validation = preprocessor.transform(
        validation_features
    )

    X_test = preprocessor.transform(
        test_features
    )

    if not (
        X_train.shape[1]
        == X_validation.shape[1]
        == X_test.shape[1]
    ):
        raise ValueError(
            f"{name}: transformed feature counts differ."
        )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation,
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test,
    )

    return {
        "variant": name,
        "predictor_count": train_features.shape[1],
        "transformed_feature_count": X_train.shape[1],
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_r2": validation_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }


def main() -> None:
    """Run all feature-ablation experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    logger.info(
        "Starting XGBoost feature-ablation experiment."
    )

    train_data = load_dataset(TRAIN_PATH)
    validation_data = load_dataset(VALIDATION_PATH)
    test_data = load_dataset(TEST_PATH)

    X_train, y_train = prepare_dataset(train_data)
    X_validation, y_validation = prepare_dataset(
        validation_data
    )
    X_test, y_test = prepare_dataset(test_data)

    validate_feature_columns(
        X_train,
        X_validation,
        X_test,
    )

    variants = {
        "full_features": [],
        "without_disposable_income": [
            "disposable_income",
        ],
        "without_financial_ratios": FINANCIAL_RATIO_FEATURES,
        "raw_features_only": ENGINEERED_FEATURES,
    }

    results = []

    for name, features_to_remove in variants.items():
        train_variant_features = remove_features(
            X_train,
            features_to_remove,
        )

        validation_variant_features = remove_features(
            X_validation,
            features_to_remove,
        )

        test_variant_features = remove_features(
            X_test,
            features_to_remove,
        )

        result = train_variant(
            name=name,
            train_features=train_variant_features,
            validation_features=validation_variant_features,
            test_features=test_variant_features,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )

        results.append(result)

    results_dataframe = pd.DataFrame(results)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / "xgboost_feature_ablation.csv"
    )

    results_dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 90)
    print("XGBOOST FEATURE ABLATION RESULTS")
    print("=" * 90)
    print(
        results_dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("=" * 90)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()