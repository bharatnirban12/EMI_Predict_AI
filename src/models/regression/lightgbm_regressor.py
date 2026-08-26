"""
Module: lightgbm_regressor.py

Description:
    Train and evaluate a LightGBM Regressor baseline for
    maximum monthly EMI prediction.

    The preprocessing pipeline is fitted only on training data
    and reused for validation and test data.

Author:
    EMIPredict AI Project
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from lightgbm import LGBMRegressor
except OSError as e:
    if "WinError 4551" in str(e):
        LGBMRegressor = None
    else:
        raise
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.features.feature_engineering import (
    engineer_features,
)
from src.preprocessing.preprocessing_pipeline import (
    create_preprocessing_pipeline,
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "regression"
)

TRAIN_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "regression_train.csv"
)

VALIDATION_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "regression_validation.csv"
)

TEST_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "regression_test.csv"
)

MODEL_ARTIFACT_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_regressor_model.pkl"
)

PREPROCESSOR_ARTIFACT_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_regressor_preprocessor.pkl"
)

TARGET_COLUMN = "max_monthly_emi"

N_ESTIMATORS = 300
LEARNING_RATE = 0.05
NUM_LEAVES = 31
MAX_DEPTH = -1
SUBSAMPLE = 0.80
COLSAMPLE_BYTREE = 0.80
RANDOM_STATE = 42


def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load a regression dataset.

    Parameters
    ----------
    file_path : Path
        Path to the dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.

    ValueError
        If the dataset is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            f"Dataset is empty: {file_path}"
        )

    logger.info(
        "Loaded dataset %s: %s",
        file_path.name,
        dataframe.shape,
    )

    return dataframe


def prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering and separate predictors
    from the regression target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Regression dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Predictor dataframe and target series.
    """
    engineered_data = engineer_features(
        dataframe
    )

    features, target = prepare_features_and_target(
        engineered_data,
        TARGET_COLUMN,
    )

    if TARGET_COLUMN in features.columns:
        raise ValueError(
            "Target leakage detected: "
            f"'{TARGET_COLUMN}' is present in predictors."
        )

    return features, target


def evaluate_model(
    model: LGBMRegressor,
    features,
    target: pd.Series,
    dataset_name: str,
) -> dict[str, float]:
    """
    Evaluate the LightGBM regression model.

    Parameters
    ----------
    model : LGBMRegressor
        Fitted LightGBM model.

    features
        Transformed predictor matrix.

    target : pd.Series
        Actual target values.

    dataset_name : str
        Dataset name.

    Returns
    -------
    dict[str, float]
        Regression metrics.
    """
    predictions = model.predict(
        features
    )

    if not np.isfinite(
        predictions
    ).all():
        raise ValueError(
            f"{dataset_name} predictions contain "
            "NaN or infinite values."
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

    logger.info(
        "%s MAE: %.4f",
        dataset_name,
        mae,
    )

    logger.info(
        "%s RMSE: %.4f",
        dataset_name,
        rmse,
    )

    logger.info(
        "%s R2: %.4f",
        dataset_name,
        r2,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_artifact(
    artifact,
    file_path: Path,
) -> None:
    """
    Save a model or preprocessing artifact.

    Parameters
    ----------
    artifact
        Object to serialize.

    file_path : Path
        Destination path.
    """
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "wb"
    ) as file:
        pickle.dump(
            artifact,
            file,
        )

    if not file_path.exists():
        raise RuntimeError(
            f"Artifact was not created: {file_path}"
        )

    logger.info(
        "Saved artifact: %s",
        file_path,
    )


def train_lightgbm() -> dict[str, float]:
    """
    Train and evaluate the LightGBM Regressor.

    Returns
    -------
    dict[str, float]
        Validation metrics.
    """
    logger.info(
        "Starting LightGBM Regression training."
    )

    train_data = load_dataset(
        TRAIN_DATA_PATH
    )

    validation_data = load_dataset(
        VALIDATION_DATA_PATH
    )

    test_data = load_dataset(
        TEST_DATA_PATH
    )

    X_train, y_train = prepare_dataset(
        train_data
    )

    X_validation, y_validation = prepare_dataset(
        validation_data
    )

    X_test, y_test = prepare_dataset(
        test_data
    )

    logger.info(
        "Training predictors shape: %s",
        X_train.shape,
    )

    logger.info(
        "Validation predictors shape: %s",
        X_validation.shape,
    )

    logger.info(
        "Test predictors shape: %s",
        X_test.shape,
    )

    if list(X_train.columns) != list(
        X_validation.columns
    ):
        raise ValueError(
            "Training and validation predictor "
            "columns do not match."
        )

    if list(X_train.columns) != list(
        X_test.columns
    ):
        raise ValueError(
            "Training and test predictor "
            "columns do not match."
        )

    logger.info(
        "Predictor column consistency validation passed."
    )

    preprocessor = create_preprocessing_pipeline(
        X_train
    )

    logger.info(
        "Fitting preprocessing pipeline "
        "on training data."
    )

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    logger.info(
        "Transforming validation data."
    )

    X_validation_transformed = (
        preprocessor.transform(
            X_validation
        )
    )

    logger.info(
        "Transforming test data."
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    logger.info(
        "Training transformed shape: %s",
        X_train_transformed.shape,
    )

    logger.info(
        "Validation transformed shape: %s",
        X_validation_transformed.shape,
    )

    logger.info(
        "Test transformed shape: %s",
        X_test_transformed.shape,
    )

    if not (
        X_train_transformed.shape[1]
        == X_validation_transformed.shape[1]
        == X_test_transformed.shape[1]
    ):
        raise ValueError(
            "Transformed feature counts do not match."
        )

    if LGBMRegressor is None:
        logger.error(
            "LightGBM is blocked by Windows Application Control "
            "policy (WinError 4551). Skipping training."
        )
        print("\nSkipped LightGBM due to system policy block.")
        return {}

    model = LGBMRegressor(
        objective="regression",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    logger.info(
        "LightGBM configuration: "
        "n_estimators=%d, learning_rate=%.3f, "
        "num_leaves=%d, max_depth=%d, "
        "subsample=%.2f, colsample_bytree=%.2f, "
        "random_state=%d",
        N_ESTIMATORS,
        LEARNING_RATE,
        NUM_LEAVES,
        MAX_DEPTH,
        SUBSAMPLE,
        COLSAMPLE_BYTREE,
        RANDOM_STATE,
    )

    logger.info(
        "Training LightGBM Regressor."
    )

    model.fit(
        X_train_transformed,
        y_train,
    )

    logger.info(
        "LightGBM Regression training completed."
    )

    validation_metrics = evaluate_model(
        model,
        X_validation_transformed,
        y_validation,
        "Validation",
    )

    logger.info(
        "Evaluating test set for final baseline reference."
    )

    test_metrics = evaluate_model(
        model,
        X_test_transformed,
        y_test,
        "Test",
    )

    save_artifact(
        model,
        MODEL_ARTIFACT_PATH,
    )

    save_artifact(
        preprocessor,
        PREPROCESSOR_ARTIFACT_PATH,
    )

    print(
        "\n" + "=" * 60
    )
    print(
        "LIGHTGBM REGRESSION BASELINE SUMMARY"
    )
    print(
        "=" * 60
    )

    print(
        f"Training records: {len(y_train):,}"
    )

    print(
        f"Validation records: {len(y_validation):,}"
    )

    print(
        f"Test records: {len(y_test):,}"
    )

    print(
        f"Transformed features: "
        f"{X_train_transformed.shape[1]}"
    )

    print(
        f"Trees: {N_ESTIMATORS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Number of leaves: {NUM_LEAVES}"
    )

    print(
        f"Max depth: {MAX_DEPTH}"
    )

    print(
        "\nValidation Metrics:"
    )

    for metric, value in validation_metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    print(
        "\nTest Metrics:"
    )

    for metric, value in test_metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    return validation_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    train_lightgbm()