"""
Module: random_forest_model.py

Description:
    Train and evaluate a Random Forest baseline for the
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
from sklearn.ensemble import RandomForestClassifier
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
    / "random_forest_model.pkl"
)

PREPROCESSOR_ARTIFACT_PATH = (
    ARTIFACTS_DIR
    / "random_forest_preprocessor.pkl"
)

TARGET_COLUMN = "emi_eligibility"

RANDOM_STATE = 42
N_ESTIMATORS = 200


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
        Classification dataset.

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
    model: RandomForestClassifier,
    features,
    target: pd.Series,
    dataset_name: str,
) -> dict[str, float]:
    """
    Evaluate the Random Forest classifier.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted Random Forest model.

    features
        Preprocessed feature matrix.

    target : pd.Series
        True target labels.

    dataset_name : str
        Dataset label used in output.

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

    weighted_precision = precision_score(
        target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        target,
        predictions,
        average="macro",
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
        weighted_precision,
    )

    logger.info(
        "%s Weighted Recall: %.4f",
        dataset_name,
        weighted_recall,
    )

    logger.info(
        "%s Weighted F1: %.4f",
        dataset_name,
        weighted_f1,
    )

    logger.info(
        "%s Macro F1: %.4f",
        dataset_name,
        macro_f1,
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
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "macro_f1": float(
            macro_f1
        ),
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


def train_random_forest() -> dict[str, float]:
    """
    Train and evaluate the Random Forest classifier.

    Returns
    -------
    dict[str, float]
        Validation metrics.
    """
    logger.info(
        "Starting Random Forest training."
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

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info(
        "Random Forest configuration: "
        "n_estimators=%d, random_state=%d, n_jobs=-1",
        N_ESTIMATORS,
        RANDOM_STATE,
    )

    logger.info(
        "Training Random Forest model."
    )

    model.fit(
        X_train_transformed,
        y_train,
    )

    logger.info(
        "Random Forest training completed."
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
        "RANDOM FOREST BASELINE SUMMARY"
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

    train_random_forest()