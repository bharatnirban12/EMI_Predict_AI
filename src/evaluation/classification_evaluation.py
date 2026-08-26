"""
Final evaluation of the tuned LightGBM classification model.

This module evaluates an already-trained model on the held-out
classification test dataset.

No model fitting occurs in this module.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.features.feature_engineering import engineer_features
from src.preprocessing.preprocessing_pipeline import (
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification_test.csv"
)

ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "classification"
)

MODEL_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_tuned_model.pkl"
)

PREPROCESSOR_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_tuned_preprocessor.pkl"
)

LABEL_MAPPING_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_tuned_label_mapping.pkl"
)

OUTPUT_DIR = (
    ARTIFACTS_DIR
    / "evaluation"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "lightgbm_tuned_test_metrics.csv"
)

FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIR
    / "lightgbm_tuned_feature_importance.csv"
)

ERROR_ANALYSIS_PATH = (
    OUTPUT_DIR
    / "lightgbm_tuned_error_analysis.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "lightgbm_tuned_confusion_matrix.csv"
)

TARGET_COLUMN = "emi_eligibility"


def load_pickle(file_path: Path):
    """Load a serialized Python artifact."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {file_path}"
        )

    with file_path.open("rb") as file:
        artifact = pickle.load(file)

    logger.info(
        "Loaded artifact: %s",
        file_path,
    )

    return artifact


def load_test_data() -> pd.DataFrame:
    """Load the held-out classification test dataset."""
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    dataframe = pd.read_csv(
        TEST_DATA_PATH,
        low_memory=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Classification test dataset is empty."
        )

    logger.info(
        "Loaded classification test dataset: %s",
        dataframe.shape,
    )

    return dataframe


def validate_target_mapping(
    target: pd.Series,
    label_mapping: dict[str, int],
) -> None:
    """Validate that all test labels exist in the saved mapping."""
    observed_labels = set(
        target.astype(str).unique()
    )

    known_labels = set(
        label_mapping.keys()
    )

    unknown_labels = (
        observed_labels - known_labels
    )

    if unknown_labels:
        raise ValueError(
            "Test contains labels absent from "
            f"the saved mapping: {sorted(unknown_labels)}"
        )


def get_feature_names(
    preprocessor,
    transformed_count: int,
) -> list[str]:
    """
    Get transformed feature names from the fitted
    preprocessing pipeline.
    """
    if hasattr(
        preprocessor,
        "get_feature_names_out",
    ):
        feature_names = list(
            preprocessor.get_feature_names_out()
        )

        if len(feature_names) == transformed_count:
            return feature_names

    logger.warning(
        "Could not retrieve transformed feature names. "
        "Using generic feature names."
    )

    return [
        f"feature_{index}"
        for index in range(transformed_count)
    ]


def evaluate() -> None:
    """Run final tuned LightGBM classification evaluation."""
    logger.info(
        "Starting final tuned LightGBM classification evaluation."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_data = load_test_data()

    model = load_pickle(
        MODEL_PATH
    )

    preprocessor = load_pickle(
        PREPROCESSOR_PATH
    )

    label_mapping = load_pickle(
        LABEL_MAPPING_PATH
    )

    engineered_data = engineer_features(
        test_data
    )

    X_test, y_test = (
        prepare_features_and_target(
            engineered_data,
            TARGET_COLUMN,
        )
    )

    logger.info(
        "Prepared test predictors: %s",
        X_test.shape,
    )

    logger.info(
        "Prepared test target: %s",
        y_test.shape,
    )

    validate_target_mapping(
        y_test,
        label_mapping,
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    logger.info(
        "Transformed test shape: %s",
        X_test_transformed.shape,
    )

    predictions = model.predict(
        X_test_transformed
    )

    if len(predictions) != len(y_test):
        raise RuntimeError(
            "Prediction count does not match "
            "test target count."
        )

    inverse_mapping = {
        value: key
        for key, value in label_mapping.items()
    }

    y_true_labels = (
        y_test.astype(str)
    )

    y_pred_labels = pd.Series(
        predictions
    ).map(
        inverse_mapping
    )

    if y_pred_labels.isna().any():
        raise RuntimeError(
            "Model produced an unknown encoded class."
        )

    accuracy = accuracy_score(
        y_true_labels,
        y_pred_labels,
    )

    weighted_precision = precision_score(
        y_true_labels,
        y_pred_labels,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true_labels,
        y_pred_labels,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true_labels,
        y_pred_labels,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true_labels,
        y_pred_labels,
        average="macro",
        zero_division=0,
    )

    class_order = [
        inverse_mapping[index]
        for index in sorted(
            inverse_mapping
        )
    ]

    matrix = confusion_matrix(
        y_true_labels,
        y_pred_labels,
        labels=class_order,
    )

    confusion_dataframe = pd.DataFrame(
        matrix,
        index=class_order,
        columns=class_order,
    )

    confusion_dataframe.index.name = (
        "actual"
    )

    confusion_dataframe.columns.name = (
        "predicted"
    )

    confusion_dataframe.to_csv(
        CONFUSION_MATRIX_PATH
    )

    logger.info(
        "Saved confusion matrix: %s",
        CONFUSION_MATRIX_PATH,
    )

    metrics_dataframe = pd.DataFrame(
        [
            {
                "model": "lightgbm_tuned",
                "dataset": "test",
                "records": len(y_test),
                "transformed_features": (
                    X_test_transformed.shape[1]
                ),
                "accuracy": accuracy,
                "weighted_precision": (
                    weighted_precision
                ),
                "weighted_recall": (
                    weighted_recall
                ),
                "weighted_f1": weighted_f1,
                "macro_f1": macro_f1,
            }
        ]
    )

    metrics_dataframe.to_csv(
        METRICS_PATH,
        index=False,
    )

    logger.info(
        "Saved test metrics: %s",
        METRICS_PATH,
    )

    if not hasattr(
        model,
        "feature_importances_",
    ):
        raise RuntimeError(
            "Loaded model does not expose "
            "feature_importances_."
        )

    feature_importances = (
        model.feature_importances_
    )

    feature_names = get_feature_names(
        preprocessor,
        X_test_transformed.shape[1],
    )

    if len(feature_importances) != len(
        feature_names
    ):
        raise RuntimeError(
            "Feature importance count does not "
            "match transformed feature count."
        )

    feature_importance_dataframe = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": feature_importances,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_importance_dataframe[
        "rank"
    ] = (
        feature_importance_dataframe.index
        + 1
    )

    feature_importance_dataframe = (
        feature_importance_dataframe[
            [
                "rank",
                "feature",
                "importance",
            ]
        ]
    )

    feature_importance_dataframe.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    logger.info(
        "Saved feature importance: %s",
        FEATURE_IMPORTANCE_PATH,
    )

    error_analysis = test_data.copy()

    error_analysis[
        "actual_emi_eligibility"
    ] = y_true_labels.values

    error_analysis[
        "predicted_emi_eligibility"
    ] = y_pred_labels.values

    error_analysis["prediction_correct"] = (
        y_true_labels.values
        == y_pred_labels.values
    )

    error_analysis[
        "prediction_error"
    ] = (
        ~error_analysis[
            "prediction_correct"
        ]
    )

    error_analysis = (
        error_analysis[
            error_analysis[
                "prediction_error"
            ]
        ]
        .copy()
    )

    error_analysis.to_csv(
        ERROR_ANALYSIS_PATH,
        index=False,
    )

    logger.info(
        "Saved classification error analysis: %s",
        ERROR_ANALYSIS_PATH,
    )

    print()
    print("=" * 70)
    print(
        "FINAL TUNED LIGHTGBM CLASSIFICATION EVALUATION"
    )
    print("=" * 70)

    print(
        f"Test records: {len(y_test):,}"
    )

    print(
        "Transformed features: "
        f"{X_test_transformed.shape[1]}"
    )

    print()
    print("Test Metrics:")
    print("-" * 70)

    print(
        f"Accuracy:           {accuracy:.4f}"
    )

    print(
        f"Weighted Precision: {weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall:    {weighted_recall:.4f}"
    )

    print(
        f"Weighted F1:        {weighted_f1:.4f}"
    )

    print(
        f"Macro F1:           {macro_f1:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print("-" * 70)

    print(
        confusion_dataframe.to_string()
    )

    print()
    print("Top 15 Features:")
    print("-" * 70)

    print(
        feature_importance_dataframe
        .head(15)
        .to_string(index=False)
    )

    print()
    print("Error Analysis:")
    print("-" * 70)

    print(
        f"Misclassified records: "
        f"{len(error_analysis):,}"
    )

    print(
        f"Correct records: "
        f"{len(test_data) - len(error_analysis):,}"
    )

    print()
    print("Evaluation Outputs:")
    print("-" * 70)

    print(
        METRICS_PATH
    )

    print(
        FEATURE_IMPORTANCE_PATH
    )

    print(
        ERROR_ANALYSIS_PATH
    )

    print(
        CONFUSION_MATRIX_PATH
    )

    logger.info(
        "Final tuned LightGBM classification "
        "evaluation completed successfully."
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

    evaluate()