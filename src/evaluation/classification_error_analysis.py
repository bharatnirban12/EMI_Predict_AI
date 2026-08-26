"""
Module: classification_error_analysis.py

Description:
    Performs detailed error analysis and explainability for the
    selected LightGBM classification model.

    The analysis includes:
        - Overall prediction errors
        - Per-class error counts
        - High_Risk error analysis
        - Confusion-pair analysis
        - LightGBM feature importance

    The saved model and preprocessing artifacts are reused.
    No model retraining is performed.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_engineering import engineer_features
from src.features.preprocessing_pipeline import (
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

ARTIFACTS_DIR = (
    PROJECT_ROOT / "artifacts" / "classification"
)

EVALUATION_DIR = (
    ARTIFACTS_DIR / "evaluation"
)

TEST_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "classification_test.csv"
)

MODEL_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_model.pkl"
)

PREPROCESSOR_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_preprocessor.pkl"
)

ERROR_ANALYSIS_PATH = (
    EVALUATION_DIR
    / "lightgbm_error_analysis.csv"
)

CLASS_SUMMARY_PATH = (
    EVALUATION_DIR
    / "lightgbm_class_error_summary.csv"
)

FEATURE_IMPORTANCE_PATH = (
    EVALUATION_DIR
    / "lightgbm_feature_importance.csv"
)

TARGET_COLUMN = "emi_eligibility"

EXPECTED_CLASSES = [
    "Eligible",
    "High_Risk",
    "Not_Eligible",
]


def load_pickle_artifact(
    path: Path,
):
    """
    Load a serialized artifact.

    Parameters
    ----------
    path : Path
        Artifact path.

    Returns
    -------
    object
        Loaded artifact.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    with path.open("rb") as file:
        artifact = pickle.load(file)

    logger.info(
        "Loaded artifact: %s",
        path,
    )

    return artifact


def load_test_data() -> pd.DataFrame:
    """
    Load the classification test dataset.

    Returns
    -------
    pd.DataFrame
        Test dataset.
    """
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: "
            f"{TEST_DATA_PATH}"
        )

    dataframe = pd.read_csv(
        TEST_DATA_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "Classification test dataset is empty."
        )

    logger.info(
        "Loaded test dataset: %s",
        dataframe.shape,
    )

    return dataframe


def prepare_test_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply project feature engineering and separate
    predictors from target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Test dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Predictors and target.
    """
    engineered_data = engineer_features(
        dataframe
    )

    features, target = (
        prepare_features_and_target(
            engineered_data,
            TARGET_COLUMN,
        )
    )

    if TARGET_COLUMN in features.columns:
        raise ValueError(
            "Target leakage detected: "
            f"{TARGET_COLUMN} is present "
            "in predictors."
        )

    logger.info(
        "Prepared predictors: %s",
        features.shape,
    )

    logger.info(
        "Prepared target: %s",
        target.shape,
    )

    return features, target


def get_feature_names(
    preprocessor,
    transformed_data,
) -> list[str]:
    """
    Retrieve transformed feature names.

    Parameters
    ----------
    preprocessor : object
        Saved preprocessing artifact.

    transformed_data : ndarray
        Transformed test data.

    Returns
    -------
    list[str]
        Feature names.
    """
    if hasattr(
        preprocessor,
        "get_feature_names_out",
    ):
        names = (
            preprocessor
            .get_feature_names_out()
        )

        return names.tolist()

    if isinstance(
        preprocessor,
        dict,
    ):
        for key in [
            "preprocessor",
            "pipeline",
            "transformer",
        ]:
            candidate = preprocessor.get(
                key
            )

            if hasattr(
                candidate,
                "get_feature_names_out",
            ):
                names = (
                    candidate
                    .get_feature_names_out()
                )

                return names.tolist()

        for key in [
            "feature_names",
            "transformed_feature_names",
        ]:
            candidate = preprocessor.get(
                key
            )

            if candidate is not None:
                return list(candidate)

    return [
        f"feature_{index}"
        for index in range(
            transformed_data.shape[1]
        )
    ]


def transform_features(
    preprocessor,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Transform predictors using the saved preprocessing artifact.

    Parameters
    ----------
    preprocessor : object
        Saved preprocessing artifact.

    features : pd.DataFrame
        Predictor dataframe.

    Returns
    -------
    np.ndarray
        Transformed feature matrix.
    """
    if hasattr(
        preprocessor,
        "transform",
    ):
        transformed = (
            preprocessor.transform(
                features
            )
        )

        return np.asarray(
            transformed
        )

    if isinstance(
        preprocessor,
        dict,
    ):
        for key in [
            "preprocessor",
            "pipeline",
            "transformer",
        ]:
            candidate = preprocessor.get(
                key
            )

            if hasattr(
                candidate,
                "transform",
            ):
                transformed = (
                    candidate.transform(
                        features
                    )
                )

                return np.asarray(
                    transformed
                )

    raise TypeError(
        "Unable to locate a preprocessing "
        "object exposing transform(). "
        f"Artifact type: "
        f"{type(preprocessor)}"
    )


def decode_predictions(
    predictions: np.ndarray,
    model,
) -> np.ndarray:
    """
    Convert model predictions into class labels.

    Parameters
    ----------
    predictions : np.ndarray
        Raw predictions.

    model : object
        Loaded LightGBM model.

    Returns
    -------
    np.ndarray
        String class labels.
    """
    predictions = np.asarray(
        predictions
    )

    if (
        predictions.dtype.kind
        in {"U", "S", "O"}
    ):
        return predictions.astype(
            str
        )

    if hasattr(
        model,
        "classes_",
    ):
        classes = np.asarray(
            model.classes_
        )

        return classes[
            predictions.astype(int)
        ]

    if isinstance(
        model,
        dict,
    ):
        class_mapping = model.get(
            "label_mapping"
        )

        if class_mapping:
            inverse_mapping = {
                value: key
                for key, value
                in class_mapping.items()
            }

            return np.array(
                [
                    inverse_mapping[
                        int(value)
                    ]
                    for value in predictions
                ]
            )

    default_mapping = {
        0: "Eligible",
        1: "High_Risk",
        2: "Not_Eligible",
    }

    return np.array(
        [
            default_mapping.get(
                int(value),
                str(value),
            )
            for value in predictions
        ]
    )


def build_error_analysis(
    original_data: pd.DataFrame,
    target: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Build detailed classification error analysis.

    Parameters
    ----------
    original_data : pd.DataFrame
        Original test records.

    target : pd.Series
        Actual target labels.

    predictions : np.ndarray
        Predicted labels.

    Returns
    -------
    pd.DataFrame
        Detailed error-analysis dataframe.
    """
    analysis = original_data.copy()

    analysis[
        "actual_class"
    ] = target.to_numpy()

    analysis[
        "predicted_class"
    ] = predictions

    analysis[
        "prediction_correct"
    ] = (
        analysis["actual_class"]
        == analysis["predicted_class"]
    )

    analysis[
        "error_type"
    ] = np.where(
        analysis[
            "prediction_correct"
        ],
        "Correct",
        (
            analysis[
                "actual_class"
            ].astype(str)
            + " -> "
            + analysis[
                "predicted_class"
            ].astype(str)
        ),
    )

    analysis[
        "actual_high_risk"
    ] = (
        analysis["actual_class"]
        == "High_Risk"
    )

    analysis[
        "predicted_high_risk"
    ] = (
        analysis["predicted_class"]
        == "High_Risk"
    )

    return analysis


def build_class_summary(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-class classification error summary.

    Parameters
    ----------
    analysis : pd.DataFrame
        Detailed error analysis.

    Returns
    -------
    pd.DataFrame
        Per-class summary.
    """
    rows = []

    total_records = len(
        analysis
    )

    for class_name in EXPECTED_CLASSES:
        actual_mask = (
            analysis["actual_class"]
            == class_name
        )

        predicted_mask = (
            analysis["predicted_class"]
            == class_name
        )

        true_positive = (
            actual_mask
            & predicted_mask
        ).sum()

        false_negative = (
            actual_mask
            & ~predicted_mask
        ).sum()

        false_positive = (
            ~actual_mask
            & predicted_mask
        ).sum()

        actual_count = (
            actual_mask.sum()
        )

        predicted_count = (
            predicted_mask.sum()
        )

        recall = (
            true_positive
            / actual_count
            if actual_count > 0
            else 0.0
        )

        precision = (
            true_positive
            / predicted_count
            if predicted_count > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        rows.append(
            {
                "class": class_name,
                "total_test_records": total_records,
                "actual_count": int(
                    actual_count
                ),
                "predicted_count": int(
                    predicted_count
                ),
                "true_positive": int(
                    true_positive
                ),
                "false_negative": int(
                    false_negative
                ),
                "false_positive": int(
                    false_positive
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(f1),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_confusion_error_summary(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize all incorrect prediction pairs.

    Parameters
    ----------
    analysis : pd.DataFrame
        Error-analysis dataframe.

    Returns
    -------
    pd.DataFrame
        Confusion-pair summary.
    """
    errors = analysis[
        ~analysis[
            "prediction_correct"
        ]
    ]

    if errors.empty:
        return pd.DataFrame(
            columns=[
                "actual_class",
                "predicted_class",
                "error_count",
                "error_percentage",
            ]
        )

    summary = (
        errors.groupby(
            [
                "actual_class",
                "predicted_class",
            ]
        )
        .size()
        .reset_index(
            name="error_count"
        )
    )

    total_errors = len(
        errors
    )

    summary[
        "error_percentage"
    ] = (
        summary["error_count"]
        / total_errors
        * 100
    )

    return summary.sort_values(
        "error_count",
        ascending=False,
    )


def extract_feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract LightGBM feature importance.

    Parameters
    ----------
    model : object
        Loaded LightGBM model.

    feature_names : list[str]
        Transformed feature names.

    Returns
    -------
    pd.DataFrame
        Feature importance table.
    """
    estimator = model

    if isinstance(
        model,
        dict,
    ):
        for key in [
            "model",
            "estimator",
            "classifier",
        ]:
            if key in model:
                estimator = model[
                    key
                ]
                break

    if not hasattr(
        estimator,
        "feature_importances_",
    ):
        raise AttributeError(
            "Loaded LightGBM model does not "
            "expose feature_importances_."
        )

    importance = np.asarray(
        estimator.feature_importances_
    )

    if len(importance) != len(
        feature_names
    ):
        raise ValueError(
            "Feature importance count does "
            "not match transformed feature "
            "count. "
            f"Importance count: "
            f"{len(importance)}, "
            f"Feature count: "
            f"{len(feature_names)}"
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    total_importance = (
        result["importance"].sum()
    )

    if total_importance > 0:
        result[
            "importance_percentage"
        ] = (
            result["importance"]
            / total_importance
            * 100
        )
    else:
        result[
            "importance_percentage"
        ] = 0.0

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )


def save_output(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save analysis output as CSV.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Output dataframe.

    path : Path
        Destination path.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Saved output: %s",
        path,
    )


def run_error_analysis() -> None:
    """
    Execute complete classification error analysis.
    """
    logger.info(
        "Starting LightGBM classification "
        "error analysis."
    )

    test_data = load_test_data()

    model = load_pickle_artifact(
        MODEL_PATH
    )

    preprocessor_artifact = load_pickle_artifact(
        PREPROCESSOR_PATH
    )

    preprocessor = preprocessor_artifact["preprocessor"]
    label_mapping = preprocessor_artifact["label_mapping"]

    features, target = (
        prepare_test_data(
            test_data
        )
    )

    transformed = transform_features(
        preprocessor,
        features,
    )

    logger.info(
        "Transformed test shape: %s",
        transformed.shape,
    )

    predictions = model.predict(
        transformed
    )

    inverse_mapping = {
        value: key
        for key, value in label_mapping.items()
    }

    predictions = np.array([
        inverse_mapping[pred]
        for pred in predictions
    ])

    predictions = np.asarray(
        predictions
    ).astype(str)

    if len(predictions) != len(
        target
    ):
        raise ValueError(
            "Prediction count does not match "
            "test target count."
        )

    unexpected = set(
        predictions
    ) - set(
        EXPECTED_CLASSES
    )

    if unexpected:
        raise ValueError(
            "Unexpected prediction classes: "
            f"{unexpected}"
        )

    analysis = build_error_analysis(
        test_data,
        target,
        predictions,
    )

    class_summary = build_class_summary(
        analysis
    )

    confusion_errors = (
        build_confusion_error_summary(
            analysis
        )
    )

    feature_names = get_feature_names(
        preprocessor,
        transformed,
    )

    feature_importance = (
        extract_feature_importance(
            model,
            feature_names,
        )
    )

    save_output(
        analysis,
        ERROR_ANALYSIS_PATH,
    )

    save_output(
        class_summary,
        CLASS_SUMMARY_PATH,
    )

    save_output(
        feature_importance,
        FEATURE_IMPORTANCE_PATH,
    )

    accuracy = (
        analysis[
            "prediction_correct"
        ].mean()
    )

    high_risk_actual = (
        analysis["actual_class"]
        == "High_Risk"
    )

    high_risk_predicted = (
        analysis["predicted_class"]
        == "High_Risk"
    )

    high_risk_tp = (
        high_risk_actual
        & high_risk_predicted
    ).sum()

    high_risk_total = (
        high_risk_actual.sum()
    )

    high_risk_recall = (
        high_risk_tp
        / high_risk_total
        if high_risk_total > 0
        else 0.0
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LIGHTGBM CLASSIFICATION ERROR ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"Test records: {len(analysis):,}"
    )

    print(
        f"Correct predictions: "
        f"{analysis['prediction_correct'].sum():,}"
    )

    print(
        f"Incorrect predictions: "
        f"{(~analysis['prediction_correct']).sum():,}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        "\nHigh_Risk Analysis"
    )

    print(
        "-" * 70
    )

    print(
        f"Actual High_Risk: "
        f"{high_risk_total:,}"
    )

    print(
        f"Correct High_Risk predictions: "
        f"{high_risk_tp:,}"
    )

    print(
        f"Missed High_Risk records: "
        f"{high_risk_total - high_risk_tp:,}"
    )

    print(
        f"High_Risk recall: "
        f"{high_risk_recall:.4f}"
    )

    print(
        "\nConfusion Error Summary"
    )

    print(
        "-" * 70
    )

    if confusion_errors.empty:
        print(
            "No classification errors found."
        )
    else:
        print(
            confusion_errors.to_string(
                index=False
            )
        )

    print(
        "\nPer-Class Summary"
    )

    print(
        "-" * 70
    )

    print(
        class_summary.to_string(
            index=False
        )
    )

    print(
        "\nTop 20 Features"
    )

    print(
        "-" * 70
    )

    print(
        feature_importance.head(
            20
        ).to_string(
            index=False
        )
    )

    print(
        "\nOutput Files"
    )

    print(
        "-" * 70
    )

    print(
        f"Error analysis: "
        f"{ERROR_ANALYSIS_PATH}"
    )

    print(
        f"Class summary: "
        f"{CLASS_SUMMARY_PATH}"
    )

    print(
        f"Feature importance: "
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    logger.info(
        "LightGBM classification error "
        "analysis completed."
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

    run_error_analysis()