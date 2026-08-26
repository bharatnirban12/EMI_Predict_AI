"""
Module: logistic_regression_model.py

Description:
    Train and evaluate a Logistic Regression baseline for the
    EMI eligibility classification task.

    The preprocessing pipeline is fitted only on the training
    dataset and reused for validation and test data.

Author:
    EMIPredict AI Project
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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
    / "classification"
)

TRAIN_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "classification_train.csv"
)

VALIDATION_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "classification_validation.csv"
)

TEST_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "classification_test.csv"
)

MODEL_ARTIFACT_PATH = (
    ARTIFACTS_DIR
    / "logistic_regression_model.pkl"
)

PREPROCESSOR_ARTIFACT_PATH = (
    ARTIFACTS_DIR
    / "classification_preprocessor.pkl"
)

TARGET_COLUMN = "emi_eligibility"

RANDOM_STATE = 42
MAX_ITER = 1000


def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load a classification dataset.

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
    Apply deterministic feature engineering and separate
    predictors from the classification target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw processed classification split.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and target vector.
    """
    engineered_data = engineer_features(
        dataframe
    )

    features, target = prepare_features_and_target(
        engineered_data,
        TARGET_COLUMN,
    )

    return features, target


def evaluate_model(
    model: LogisticRegression,
    features,
    target: pd.Series,
    dataset_name: str,
) -> dict[str, float]:
    """
    Evaluate the classification model.

    Parameters
    ----------
    model : LogisticRegression
        Fitted classification model.

    features
        Preprocessed feature matrix.

    target : pd.Series
        True target labels.

    dataset_name : str
        Name used in logging.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """
    predictions = model.predict(
        features
    )

    accuracy = accuracy_score(
        target,
        predictions,
    )

    precision = precision_score(
        target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    logger.info(
        "%s Accuracy: %.4f",
        dataset_name,
        accuracy,
    )

    logger.info(
        "%s Weighted Precision: %.4f",
        dataset_name,
        precision,
    )

    logger.info(
        "%s Weighted Recall: %.4f",
        dataset_name,
        recall,
    )

    logger.info(
        "%s Weighted F1: %.4f",
        dataset_name,
        f1,
    )

    print(
        f"\n{dataset_name} Classification Report"
    )
    print("=" * 60)
    print(
        classification_report(
            target,
            predictions,
            zero_division=0,
        )
    )

    print(
        f"{dataset_name} Confusion Matrix"
    )
    print("=" * 60)
    print(
        confusion_matrix(
            target,
            predictions,
        )
    )

    return {
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
    }


def save_artifact(
    artifact,
    file_path: Path,
) -> None:
    """
    Save a Python object using pickle.

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


def train_logistic_regression() -> dict[str, float]:
    """
    Train and evaluate the Logistic Regression baseline.

    Returns
    -------
    dict[str, float]
        Validation metrics.
    """
    logger.info(
        "Starting Logistic Regression training."
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

    preprocessor = create_preprocessing_pipeline(
        X_train
    )

    logger.info(
        "Fitting preprocessing pipeline on training data."
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

    model = LogisticRegression(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Training Logistic Regression model."
    )

    model.fit(
        X_train_transformed,
        y_train,
    )

    logger.info(
        "Logistic Regression training completed."
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
        "LOGISTIC REGRESSION BASELINE SUMMARY"
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

    train_logistic_regression()