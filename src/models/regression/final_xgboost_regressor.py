"""
Train the final tuned XGBoost regression model for maximum monthly EMI.

The model predicts:
    max_monthly_emi

The preprocessing pipeline is fitted exclusively on training data and then
reused for validation and test data to prevent data leakage.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from src.features.feature_engineering import engineer_features
from src.preprocessing.preprocessing_pipeline import (
    create_preprocessing_pipeline,
    prepare_features_and_target,
)

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "max_monthly_emi"

TRAIN_PATH = Path("data/processed/regression_train.csv")
VALIDATION_PATH = Path("data/processed/regression_validation.csv")
TEST_PATH = Path("data/processed/regression_test.csv")

ARTIFACT_DIRECTORY = Path("artifacts/regression")

MODEL_PATH = ARTIFACT_DIRECTORY / "xgboost_regressor_model.pkl"
PREPROCESSOR_PATH = ARTIFACT_DIRECTORY / "xgboost_regressor_preprocessor.pkl"

RANDOM_STATE = 42

XGBOOST_PARAMETERS = {
    "n_estimators": 400,
    "max_depth": 7,
    "learning_rate": 0.08,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load a processed regression dataset.

    Parameters
    ----------
    path:
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset was not found: {path}"
        )

    dataframe = pd.read_csv(path, low_memory=False)

    LOGGER.info(
        "Loaded %s: %s",
        path.name,
        dataframe.shape,
    )

    return dataframe


def prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering and prepare predictors/target.

    Parameters
    ----------
    dataframe:
        Raw processed regression dataframe.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Predictor dataframe and target series.
    """
    engineered = engineer_features(dataframe)

    predictors, target = prepare_features_and_target(
        engineered,
        TARGET_COLUMN,
    )

    return predictors, target


def calculate_metrics(
    actual: pd.Series,
    predicted,
) -> dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Parameters
    ----------
    actual:
        Actual target values.

    predicted:
        Model predictions.

    Returns
    -------
    dict[str, float]
        MAE, RMSE, and R².
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(
        actual,
        predicted,
    ))
    r2 = r2_score(actual, predicted)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_artifact(
    artifact,
    path: Path,
) -> None:
    """
    Serialize an artifact to disk.

    Parameters
    ----------
    artifact:
        Object to serialize.

    path:
        Output path.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("wb") as file:
        pickle.dump(artifact, file)

    LOGGER.info(
        "Saved artifact: %s",
        path.resolve(),
    )


def main() -> None:
    """Train and evaluate the final tuned XGBoost model."""
    configure_logging()

    LOGGER.info(
        "Starting final tuned XGBoost regression training."
    )

    train_dataframe = load_dataset(TRAIN_PATH)
    validation_dataframe = load_dataset(VALIDATION_PATH)
    test_dataframe = load_dataset(TEST_PATH)

    X_train, y_train = prepare_dataset(train_dataframe)
    X_validation, y_validation = prepare_dataset(
        validation_dataframe
    )
    X_test, y_test = prepare_dataset(test_dataframe)

    LOGGER.info(
        "Training predictors: %s",
        X_train.shape,
    )
    LOGGER.info(
        "Validation predictors: %s",
        X_validation.shape,
    )
    LOGGER.info(
        "Test predictors: %s",
        X_test.shape,
    )

    if TARGET_COLUMN in X_train.columns:
        raise ValueError(
            f"Target leakage detected: {TARGET_COLUMN} "
            "exists in training predictors."
        )

    if TARGET_COLUMN in X_validation.columns:
        raise ValueError(
            f"Target leakage detected: {TARGET_COLUMN} "
            "exists in validation predictors."
        )

    if TARGET_COLUMN in X_test.columns:
        raise ValueError(
            f"Target leakage detected: {TARGET_COLUMN} "
            "exists in test predictors."
        )

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation predictor columns do not match."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and test predictor columns do not match."
        )

    LOGGER.info(
        "Creating preprocessing pipeline."
    )

    preprocessor = create_preprocessing_pipeline(
        X_train
    )

    LOGGER.info(
        "Fitting preprocessing pipeline on training data only."
    )

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    LOGGER.info(
        "Transforming validation data."
    )

    X_validation_transformed = preprocessor.transform(
        X_validation
    )

    LOGGER.info(
        "Transforming test data."
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    LOGGER.info(
        "Training transformed shape: %s",
        X_train_transformed.shape,
    )

    LOGGER.info(
        "Validation transformed shape: %s",
        X_validation_transformed.shape,
    )

    LOGGER.info(
        "Test transformed shape: %s",
        X_test_transformed.shape,
    )

    LOGGER.info(
        "Training final XGBoost model with parameters: %s",
        XGBOOST_PARAMETERS,
    )

    model = XGBRegressor(
        **XGBOOST_PARAMETERS
    )

    model.fit(
        X_train_transformed,
        y_train,
        verbose=False,
    )

    LOGGER.info(
        "Final XGBoost model training completed."
    )

    validation_predictions = model.predict(
        X_validation_transformed
    )

    test_predictions = model.predict(
        X_test_transformed
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    save_artifact(
        model,
        MODEL_PATH,
    )

    save_artifact(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    print()
    print("=" * 70)
    print("FINAL TUNED XGBOOST REGRESSION")
    print("=" * 70)

    print(f"Training records: {len(y_train):,}")
    print(f"Validation records: {len(y_validation):,}")
    print(f"Test records: {len(y_test):,}")

    print(
        f"Transformed features: "
        f"{X_train_transformed.shape[1]}"
    )

    print()
    print("Hyperparameters:")
    print("-" * 70)

    for parameter, value in XGBOOST_PARAMETERS.items():
        print(f"{parameter}: {value}")

    print()
    print("Validation Metrics:")
    print("-" * 70)
    print(
        f"MAE:  {validation_metrics['mae']:.4f}"
    )
    print(
        f"RMSE: {validation_metrics['rmse']:.4f}"
    )
    print(
        f"R²:   {validation_metrics['r2']:.4f}"
    )

    print()
    print("Test Metrics:")
    print("-" * 70)
    print(
        f"MAE:  {test_metrics['mae']:.4f}"
    )
    print(
        f"RMSE: {test_metrics['rmse']:.4f}"
    )
    print(
        f"R²:   {test_metrics['r2']:.4f}"
    )

    print()
    print("Requirement Validation:")
    print("-" * 70)

    mae_pass = test_metrics["mae"] < 500
    r2_pass = test_metrics["r2"] > 0.85

    print(
        f"MAE < 500: "
        f"{'PASS' if mae_pass else 'FAIL'}"
    )
    print(
        f"R² > 0.85: "
        f"{'PASS' if r2_pass else 'FAIL'}"
    )

    print()
    print("Artifacts:")
    print("-" * 70)
    print(MODEL_PATH.resolve())
    print(PREPROCESSOR_PATH.resolve())

    print("=" * 70)

    if not mae_pass or not r2_pass:
        raise RuntimeError(
            "Final XGBoost model failed the required "
            "regression quality thresholds."
        )

    LOGGER.info(
        "Final tuned XGBoost regression completed successfully."
    )


if __name__ == "__main__":
    main()