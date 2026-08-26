"""
Module: verify_regression_preprocessing.py

Description:
    Verify the regression feature-engineering and preprocessing
    pipeline before regression model training.

    This script validates:
    - regression dataset loading
    - feature engineering
    - target separation
    - target leakage prevention
    - preprocessing consistency
    - transformed feature shapes
    - transformed missing values
    - target shapes and statistics

Author:
    EMIPredict AI Project
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    engineer_features,
)
from src.preprocessing.preprocessing_pipeline import (
    create_preprocessing_pipeline,
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_PATH = (
    PROCESSED_DATA_DIR
    / "regression_train.csv"
)

VALIDATION_PATH = (
    PROCESSED_DATA_DIR
    / "regression_validation.csv"
)

TEST_PATH = (
    PROCESSED_DATA_DIR
    / "regression_test.csv"
)

TARGET_COLUMN = "max_monthly_emi"


def load_dataset(
    path: Path,
) -> pd.DataFrame:
    """
    Load a regression dataset.

    Parameters
    ----------
    path : Path
        Dataset path.

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


def prepare_regression_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering and separate predictors
    from the regression target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw regression split.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Predictors and regression target.
    """
    engineered_data = engineer_features(
        dataframe
    )

    features, target = prepare_features_and_target(
        engineered_data,
        TARGET_COLUMN,
    )

    return features, target


def verify_target(
    target: pd.Series,
    dataset_name: str,
) -> None:
    """
    Verify regression target integrity.

    Parameters
    ----------
    target : pd.Series
        Regression target.

    dataset_name : str
        Dataset name.
    """
    if target.empty:
        raise ValueError(
            f"{dataset_name} target is empty."
        )

    if target.isna().any():
        raise ValueError(
            f"{dataset_name} target contains "
            "missing values."
        )

    if not pd.api.types.is_numeric_dtype(
        target
    ):
        raise TypeError(
            f"{dataset_name} target is not numeric. "
            f"Dtype: {target.dtype}"
        )

    if not np.isfinite(
        target.to_numpy()
    ).all():
        raise ValueError(
            f"{dataset_name} target contains "
            "non-finite values."
        )


def verify_no_target_leakage(
    features: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Verify that the regression target is absent
    from the predictor dataframe.

    Parameters
    ----------
    features : pd.DataFrame
        Predictor dataframe.

    dataset_name : str
        Dataset name.
    """
    if TARGET_COLUMN in features.columns:
        raise ValueError(
            f"TARGET LEAKAGE detected in "
            f"{dataset_name}: "
            f"'{TARGET_COLUMN}' is present "
            "in predictors."
        )

    logger.info(
        "%s target leakage validation passed.",
        dataset_name,
    )


def verify_transformed_data(
    transformed,
    dataset_name: str,
) -> None:
    """
    Verify transformed feature matrix.

    Parameters
    ----------
    transformed
        Transformed feature matrix.

    dataset_name : str
        Dataset name.
    """
    if transformed is None:
        raise ValueError(
            f"{dataset_name} transformed data is None."
        )

    if hasattr(
        transformed,
        "toarray",
    ):
        values = transformed.toarray()
    else:
        values = np.asarray(
            transformed
        )

    if values.ndim != 2:
        raise ValueError(
            f"{dataset_name} transformed data "
            f"must be 2-dimensional. "
            f"Shape: {values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{dataset_name} transformed data "
            "contains NaN or infinite values."
        )


def print_target_statistics(
    target: pd.Series,
    dataset_name: str,
) -> None:
    """
    Print regression target statistics.

    Parameters
    ----------
    target : pd.Series
        Regression target.

    dataset_name : str
        Dataset name.
    """
    statistics = target.describe()

    print(
        f"\n{dataset_name} Target Statistics"
    )
    print("=" * 60)
    print(statistics)


def main() -> None:
    """
    Execute regression preprocessing verification.
    """
    logger.info(
        "Starting regression preprocessing verification."
    )

    train_data = load_dataset(
        TRAIN_PATH
    )

    validation_data = load_dataset(
        VALIDATION_PATH
    )

    test_data = load_dataset(
        TEST_PATH
    )

    (
        X_train,
        y_train,
    ) = prepare_regression_dataset(
        train_data
    )

    (
        X_validation,
        y_validation,
    ) = prepare_regression_dataset(
        validation_data
    )

    (
        X_test,
        y_test,
    ) = prepare_regression_dataset(
        test_data
    )

    logger.info(
        "Training feature shape: %s",
        X_train.shape,
    )

    logger.info(
        "Validation feature shape: %s",
        X_validation.shape,
    )

    logger.info(
        "Test feature shape: %s",
        X_test.shape,
    )

    logger.info(
        "Training target shape: %s",
        y_train.shape,
    )

    logger.info(
        "Validation target shape: %s",
        y_validation.shape,
    )

    logger.info(
        "Test target shape: %s",
        y_test.shape,
    )

    verify_target(
        y_train,
        "Training",
    )

    verify_target(
        y_validation,
        "Validation",
    )

    verify_target(
        y_test,
        "Test",
    )

    verify_no_target_leakage(
        X_train,
        "Training",
    )

    verify_no_target_leakage(
        X_validation,
        "Validation",
    )

    verify_no_target_leakage(
        X_test,
        "Test",
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
        "Fitting regression preprocessing pipeline "
        "on training data only."
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

    verify_transformed_data(
        X_train_transformed,
        "Training",
    )

    verify_transformed_data(
        X_validation_transformed,
        "Validation",
    )

    verify_transformed_data(
        X_test_transformed,
        "Test",
    )

    train_feature_count = (
        X_train_transformed.shape[1]
    )

    validation_feature_count = (
        X_validation_transformed.shape[1]
    )

    test_feature_count = (
        X_test_transformed.shape[1]
    )

    if not (
        train_feature_count
        == validation_feature_count
        == test_feature_count
    ):
        raise ValueError(
            "Transformed feature counts do not match."
        )

    logger.info(
        "Transformed feature count consistency "
        "validation passed."
    )

    if not (
        len(X_train)
        == len(y_train)
    ):
        raise ValueError(
            "Training predictor and target row "
            "counts do not match."
        )

    if not (
        len(X_validation)
        == len(y_validation)
    ):
        raise ValueError(
            "Validation predictor and target row "
            "counts do not match."
        )

    if not (
        len(X_test)
        == len(y_test)
    ):
        raise ValueError(
            "Test predictor and target row "
            "counts do not match."
        )

    print(
        "\n" + "=" * 70
    )
    print(
        "REGRESSION PREPROCESSING VERIFICATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Target column: {TARGET_COLUMN}"
    )

    print(
        f"Training input shape: "
        f"{train_data.shape}"
    )

    print(
        f"Validation input shape: "
        f"{validation_data.shape}"
    )

    print(
        f"Test input shape: "
        f"{test_data.shape}"
    )

    print(
        f"\nTraining predictors: "
        f"{X_train.shape}"
    )

    print(
        f"Validation predictors: "
        f"{X_validation.shape}"
    )

    print(
        f"Test predictors: "
        f"{X_test.shape}"
    )

    print(
        f"\nTraining target: "
        f"{y_train.shape}"
    )

    print(
        f"Validation target: "
        f"{y_validation.shape}"
    )

    print(
        f"Test target: "
        f"{y_test.shape}"
    )

    print(
        f"\nTraining transformed: "
        f"{X_train_transformed.shape}"
    )

    print(
        f"Validation transformed: "
        f"{X_validation_transformed.shape}"
    )

    print(
        f"Test transformed: "
        f"{X_test_transformed.shape}"
    )

    print(
        "\nTarget leakage:"
    )
    print(
        f"'{TARGET_COLUMN}' in predictors: "
        f"{TARGET_COLUMN in X_train.columns}"
    )

    print(
        f"\nSame predictor columns: "
        f"{list(X_train.columns) == list(X_validation.columns) == list(X_test.columns)}"
    )

    print(
        f"Same transformed feature count: "
        f"{train_feature_count == validation_feature_count == test_feature_count}"
    )

    print(
        "\nTarget statistics:"
    )

    print_target_statistics(
        y_train,
        "Training",
    )

    print_target_statistics(
        y_validation,
        "Validation",
    )

    print_target_statistics(
        y_test,
        "Test",
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "VERIFICATION STATUS: PASS"
    )
    print(
        "=" * 70
    )

    logger.info(
        "Regression preprocessing verification completed."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    main()