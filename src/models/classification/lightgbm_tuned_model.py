"""
Train and evaluate the final tuned LightGBM multiclass classifier.

The preprocessing pipeline is fitted only on the training dataset
and reused for validation and test datasets.

This module is intentionally separate from lightgbm_model.py so that
the baseline and tuned model artifacts remain reproducible and
independently available.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.features.feature_engineering import engineer_features
from src.preprocessing.preprocessing_pipeline import (
    create_preprocessing_pipeline,
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

ARTIFACTS_DIR = (
    PROJECT_ROOT / "artifacts" / "classification"
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
    / "lightgbm_tuned_model.pkl"
)

PREPROCESSOR_ARTIFACT_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_tuned_preprocessor.pkl"
)

LABEL_MAPPING_ARTIFACT_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_tuned_label_mapping.pkl"
)

TARGET_COLUMN = "emi_eligibility"

RANDOM_STATE = 42

N_ESTIMATORS = 400
LEARNING_RATE = 0.08
NUM_LEAVES = 63
MAX_DEPTH = 10
MIN_CHILD_SAMPLES = 50
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8


def load_dataset(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load a classification dataset.

    Parameters
    ----------
    file_path : Path
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
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            f"Dataset is empty: {file_path}"
        )

    logger.info(
        "Loaded %s: %s",
        file_path.name,
        dataframe.shape,
    )

    return dataframe


def prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply project feature engineering and separate
    predictors from the target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw processed classification split.

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

    return features, target


def encode_target(
    train_target: pd.Series,
    validation_target: pd.Series,
    test_target: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    dict[str, int],
]:
    """
    Encode classification labels using a mapping learned
    exclusively from the training target.

    Parameters
    ----------
    train_target : pd.Series
        Training target.

    validation_target : pd.Series
        Validation target.

    test_target : pd.Series
        Test target.

    Returns
    -------
    tuple
        Encoded train, validation, test targets and label mapping.
    """
    classes = sorted(
        train_target.astype(str).unique()
    )

    label_mapping = {
        label: index
        for index, label in enumerate(classes)
    }

    validation_classes = set(
        validation_target.astype(str).unique()
    )

    test_classes = set(
        test_target.astype(str).unique()
    )

    unknown_validation = (
        validation_classes
        - set(label_mapping)
    )

    unknown_test = (
        test_classes
        - set(label_mapping)
    )

    if unknown_validation:
        raise ValueError(
            "Validation contains unseen target classes: "
            f"{sorted(unknown_validation)}"
        )

    if unknown_test:
        raise ValueError(
            "Test contains unseen target classes: "
            f"{sorted(unknown_test)}"
        )

    train_encoded = (
        train_target.astype(str)
        .map(label_mapping)
    )

    validation_encoded = (
        validation_target.astype(str)
        .map(label_mapping)
    )

    test_encoded = (
        test_target.astype(str)
        .map(label_mapping)
    )

    if train_encoded.isna().any():
        raise ValueError(
            "Training target encoding produced missing values."
        )

    if validation_encoded.isna().any():
        raise ValueError(
            "Validation target encoding produced missing values."
        )

    if test_encoded.isna().any():
        raise ValueError(
            "Test target encoding produced missing values."
        )

    return (
        train_encoded.astype(int),
        validation_encoded.astype(int),
        test_encoded.astype(int),
        label_mapping,
    )


def evaluate_model(
    model: LGBMClassifier,
    features,
    encoded_target: pd.Series,
    label_mapping: dict[str, int],
    dataset_name: str,
) -> dict[str, float]:
    """
    Evaluate the tuned LightGBM model.

    Parameters
    ----------
    model : LGBMClassifier
        Fitted model.

    features
        Preprocessed feature matrix.

    encoded_target : pd.Series
        Integer-encoded target.

    label_mapping : dict[str, int]
        Target-label mapping.

    dataset_name : str
        Evaluation split name.

    Returns
    -------
    dict[str, float]
        Classification metrics.
    """
    predictions = model.predict(
        features
    )

    accuracy = accuracy_score(
        encoded_target,
        predictions,
    )

    weighted_precision = precision_score(
        encoded_target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        encoded_target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        encoded_target,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        encoded_target,
        predictions,
        average="macro",
        zero_division=0,
    )

    inverse_mapping = {
        value: key
        for key, value in label_mapping.items()
    }

    ordered_labels = sorted(
        inverse_mapping
    )

    ordered_names = [
        inverse_mapping[index]
        for index in ordered_labels
    ]

    report = classification_report(
        encoded_target,
        predictions,
        labels=ordered_labels,
        target_names=ordered_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        encoded_target,
        predictions,
        labels=ordered_labels,
    )

    print(
        f"\n{dataset_name} Classification Report"
    )
    print("=" * 60)
    print(report)

    print(
        f"{dataset_name} Confusion Matrix"
    )
    print("=" * 60)
    print(matrix)

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
    Serialize an artifact using pickle.

    Parameters
    ----------
    artifact
        Python object to serialize.

    file_path : Path
        Destination artifact path.
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


def train_tuned_lightgbm() -> dict[str, float]:
    """
    Train and evaluate the final tuned LightGBM classifier.

    Returns
    -------
    dict[str, float]
        Final test metrics.
    """
    logger.info(
        "Starting final tuned LightGBM classification."
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
        "Training predictor shape: %s",
        X_train.shape,
    )

    logger.info(
        "Validation predictor shape: %s",
        X_validation.shape,
    )

    logger.info(
        "Test predictor shape: %s",
        X_test.shape,
    )

    if list(X_train.columns) != list(
        X_validation.columns
    ):
        raise ValueError(
            "Training and validation predictor columns differ."
        )

    if list(X_train.columns) != list(
        X_test.columns
    ):
        raise ValueError(
            "Training and test predictor columns differ."
        )

    preprocessor = create_preprocessing_pipeline(
        X_train
    )

    logger.info(
        "Fitting preprocessing pipeline on training data only."
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

    if (
        X_train_transformed.shape[1]
        != X_validation_transformed.shape[1]
    ):
        raise ValueError(
            "Training and validation transformed feature counts differ."
        )

    if (
        X_train_transformed.shape[1]
        != X_test_transformed.shape[1]
    ):
        raise ValueError(
            "Training and test transformed feature counts differ."
        )

    (
        y_train_encoded,
        y_validation_encoded,
        y_test_encoded,
        label_mapping,
    ) = encode_target(
        y_train,
        y_validation,
        y_test,
    )

    number_of_classes = len(
        label_mapping
    )

    if number_of_classes < 2:
        raise ValueError(
            "At least two target classes are required."
        )

    logger.info(
        "Target mapping: %s",
        label_mapping,
    )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=number_of_classes,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        min_child_samples=MIN_CHILD_SAMPLES,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    logger.info(
        "Final tuned LightGBM configuration:"
    )

    logger.info(
        "n_estimators=%d",
        N_ESTIMATORS,
    )

    logger.info(
        "learning_rate=%.2f",
        LEARNING_RATE,
    )

    logger.info(
        "num_leaves=%d",
        NUM_LEAVES,
    )

    logger.info(
        "max_depth=%d",
        MAX_DEPTH,
    )

    logger.info(
        "min_child_samples=%d",
        MIN_CHILD_SAMPLES,
    )

    logger.info(
        "subsample=%.2f",
        SUBSAMPLE,
    )

    logger.info(
        "colsample_bytree=%.2f",
        COLSAMPLE_BYTREE,
    )

    logger.info(
        "Training final tuned LightGBM model."
    )

    model.fit(
        X_train_transformed,
        y_train_encoded,
    )

    logger.info(
        "Final tuned LightGBM training completed."
    )

    validation_metrics = evaluate_model(
        model=model,
        features=X_validation_transformed,
        encoded_target=y_validation_encoded,
        label_mapping=label_mapping,
        dataset_name="Validation",
    )

    test_metrics = evaluate_model(
        model=model,
        features=X_test_transformed,
        encoded_target=y_test_encoded,
        label_mapping=label_mapping,
        dataset_name="Test",
    )

    save_artifact(
        model,
        MODEL_ARTIFACT_PATH,
    )

    save_artifact(
        preprocessor,
        PREPROCESSOR_ARTIFACT_PATH,
    )

    save_artifact(
        label_mapping,
        LABEL_MAPPING_ARTIFACT_PATH,
    )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "FINAL TUNED LIGHTGBM CLASSIFICATION"
    )
    print(
        "=" * 70
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
        "Transformed features: "
        f"{X_train_transformed.shape[1]}"
    )

    print(
        "\nHyperparameters:"
    )
    print(
        "-" * 70
    )
    print(
        f"n_estimators: {N_ESTIMATORS}"
    )
    print(
        f"learning_rate: {LEARNING_RATE}"
    )
    print(
        f"num_leaves: {NUM_LEAVES}"
    )
    print(
        f"max_depth: {MAX_DEPTH}"
    )
    print(
        f"min_child_samples: {MIN_CHILD_SAMPLES}"
    )
    print(
        f"subsample: {SUBSAMPLE}"
    )
    print(
        f"colsample_bytree: {COLSAMPLE_BYTREE}"
    )
    print(
        f"random_state: {RANDOM_STATE}"
    )

    print(
        "\nValidation Metrics:"
    )
    print(
        "-" * 70
    )

    for metric, value in validation_metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    print(
        "\nTest Metrics:"
    )
    print(
        "-" * 70
    )

    for metric, value in test_metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    print(
        "\nExpected tuning reference:"
    )
    print(
        "-" * 70
    )
    print(
        "Validation Accuracy: 0.9786"
    )
    print(
        "Validation Weighted F1: 0.9780"
    )
    print(
        "Validation Macro F1: 0.9057"
    )
    print(
        "Validation High_Risk Recall: 0.7124"
    )
    print(
        "Test Accuracy: 0.9787"
    )
    print(
        "Test Weighted F1: 0.9781"
    )
    print(
        "Test Macro F1: 0.9055"
    )
    print(
        "Test High_Risk Recall: 0.7090"
    )

    print(
        "\nArtifacts:"
    )
    print(
        "-" * 70
    )
    print(
        MODEL_ARTIFACT_PATH
    )
    print(
        PREPROCESSOR_ARTIFACT_PATH
    )
    print(
        LABEL_MAPPING_ARTIFACT_PATH
    )

    logger.info(
        "Final tuned LightGBM classification completed."
    )

    return test_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    train_tuned_lightgbm()