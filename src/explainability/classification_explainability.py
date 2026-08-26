"""
Module: classification_explainability.py

Description:
    Generate global SHAP-based feature importance for the
    selected LightGBM classification model.

    The saved LightGBM model and preprocessing artifact are
    reused. No model retraining is performed.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "SHAP is required for classification explainability. "
        "Install it with: pip install shap"
    ) from exc

from src.features.feature_engineering import (
    engineer_features,
)
from src.preprocessing.preprocessing_pipeline import (
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

CLASSIFICATION_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "classification"
)

EXPLAINABILITY_DIR = (
    CLASSIFICATION_ARTIFACT_DIR
    / "explainability"
)

TEST_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "classification_test.csv"
)

MODEL_PATH = (
    CLASSIFICATION_ARTIFACT_DIR
    / "lightgbm_tuned_model.pkl"
)

PREPROCESSOR_PATH = (
    CLASSIFICATION_ARTIFACT_DIR
    / "lightgbm_tuned_preprocessor.pkl"
)

LABEL_MAPPING_PATH = (
    CLASSIFICATION_ARTIFACT_DIR
    / "lightgbm_tuned_label_mapping.pkl"
)

OUTPUT_PATH = (
    EXPLAINABILITY_DIR
    / "lightgbm_tuned_shap_feature_importance.csv"
)

TARGET_COLUMN = "emi_eligibility"

TOP_N = 30

HIGH_RISK_CLASS = "High_Risk"

HIGH_RISK_ERROR_SHAP_OUTPUT = (
    EXPLAINABILITY_DIR
    / "lightgbm_tuned_high_risk_error_shap.csv"
)

HIGH_RISK_ERROR_SUMMARY_OUTPUT = (
    EXPLAINABILITY_DIR
    / "lightgbm_tuned_high_risk_error_summary.csv"
)


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
    Load classification test data.

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


def prepare_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply feature engineering and prepare predictors.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original test dataset.

    Returns
    -------
    pd.DataFrame
        Predictor dataframe.
    """
    engineered_data = engineer_features(
        dataframe
    )

    features, _ = (
        prepare_features_and_target(
            engineered_data,
            TARGET_COLUMN,
        )
    )

    if TARGET_COLUMN in features.columns:
        raise ValueError(
            "Target leakage detected: "
            f"{TARGET_COLUMN} exists in predictors."
        )

    logger.info(
        "Prepared predictors: %s",
        features.shape,
    )

    return features


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
        for key in (
            "preprocessor",
            "pipeline",
            "transformer",
        ):
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
        "Could not find a preprocessing "
        "object exposing transform(). "
        f"Artifact type: "
        f"{type(preprocessor)}"
    )


def get_feature_names(
    preprocessor,
    transformed: np.ndarray,
) -> list[str]:
    """
    Obtain transformed feature names.

    Parameters
    ----------
    preprocessor : object
        Saved preprocessing artifact.

    transformed : np.ndarray
        Transformed feature matrix.

    Returns
    -------
    list[str]
        Feature names.
    """
    if hasattr(
        preprocessor,
        "get_feature_names_out",
    ):
        return (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

    if isinstance(
        preprocessor,
        dict,
    ):
        for key in (
            "preprocessor",
            "pipeline",
            "transformer",
        ):
            candidate = preprocessor.get(
                key
            )

            if hasattr(
                candidate,
                "get_feature_names_out",
            ):
                return (
                    candidate
                    .get_feature_names_out()
                    .tolist()
                )

        for key in (
            "feature_names",
            "transformed_feature_names",
        ):
            candidate = preprocessor.get(
                key
            )

            if candidate is not None:
                return list(candidate)

    return [
        f"feature_{index}"
        for index in range(
            transformed.shape[1]
        )
    ]


def get_model_from_artifact(
    model_artifact,
):
    """
    Extract the estimator from a serialized artifact.

    Parameters
    ----------
    model_artifact : object
        Loaded model artifact.

    Returns
    -------
    object
        Model estimator.
    """
    if isinstance(
        model_artifact,
        dict,
    ):
        for key in (
            "model",
            "estimator",
            "classifier",
        ):
            if key in model_artifact:
                return model_artifact[key]

    return model_artifact


def normalize_predictions(
    predictions: np.ndarray,
    label_mapping,
) -> np.ndarray:
    """
    Convert model prediction values into original class labels.

    Parameters
    ----------
    predictions : np.ndarray
        Raw model predictions.

    label_mapping : dict
        Mapping from encoded class values to original labels.

    Returns
    -------
    np.ndarray
        Normalized string class labels.
    """
    predictions = np.asarray(predictions)

    if predictions.dtype.kind in {"U", "S", "O"}:
        return predictions.astype(str)

    if not isinstance(label_mapping, dict):
        raise TypeError(
            "Expected label mapping artifact to be a dictionary. "
            f"Received: {type(label_mapping)}"
        )

    normalized_mapping = {}

    for key, value in label_mapping.items():
        try:
            normalized_mapping[int(key)] = str(value)
        except (TypeError, ValueError):
            try:
                normalized_mapping[int(value)] = str(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid label mapping entry: "
                    f"{key!r}: {value!r}"
                ) from exc

    normalized_predictions = []

    for prediction in predictions:
        prediction_int = int(prediction)

        if prediction_int not in normalized_mapping:
            raise ValueError(
                "Prediction value is missing from label mapping: "
                f"{prediction_int}. "
                f"Available mappings: {normalized_mapping}"
            )

        normalized_predictions.append(
            normalized_mapping[prediction_int]
        )

    return np.asarray(
        normalized_predictions,
        dtype=str,
    )


def get_predictions_and_class_labels(
    model,
    transformed: np.ndarray,
    test_data: pd.DataFrame,
    label_mapping,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate model predictions and normalize them
    to the original target labels.
    """
    if not hasattr(model, "predict"):
        raise TypeError(
            "Loaded classification model does not "
            "provide predict()."
        )

    raw_predictions = np.asarray(
        model.predict(transformed)
    )

    predictions = normalize_predictions(
        raw_predictions,
        label_mapping,
    )

    actual = test_data[
        TARGET_COLUMN
    ].astype(str).to_numpy()

    if len(predictions) != len(actual):
        raise ValueError(
            "Prediction count does not match "
            "test-record count. "
            f"Predictions: {len(predictions)}, "
            f"Actual: {len(actual)}"
        )

    expected_classes = {
        "Eligible",
        "High_Risk",
        "Not_Eligible",
    }

    actual_classes = set(actual)
    predicted_classes = set(predictions)

    if not actual_classes.issubset(expected_classes):
        raise ValueError(
            "Unexpected actual labels: "
            f"{actual_classes}"
        )

    if not predicted_classes.issubset(expected_classes):
        raise ValueError(
            "Unexpected predicted labels: "
            f"{predicted_classes}"
        )

    return predictions, actual

def calculate_shap_importance(
    model,
    transformed: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calculate global mean absolute SHAP importance.

    Parameters
    ----------
    model : object
        LightGBM estimator.

    transformed : np.ndarray
        Transformed test data.

    feature_names : list[str]
        Transformed feature names.

    Returns
    -------
    pd.DataFrame
        Feature importance dataframe.
    """
    logger.info(
        "Creating SHAP TreeExplainer."
    )

    explainer = shap.TreeExplainer(
        model
    )

    logger.info(
        "Calculating SHAP values."
    )

    shap_values = explainer.shap_values(
        transformed
    )

    if isinstance(
        shap_values,
        list,
    ):
        stacked_values = np.stack(
            [
                np.asarray(values)
                for values in shap_values
            ],
            axis=0,
        )

        absolute_values = np.abs(
            stacked_values
        )

        mean_absolute_values = (
            absolute_values.mean(
                axis=(0, 1)
            )
        )

    else:
        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 2:
            mean_absolute_values = (
                np.abs(shap_array)
                .mean(axis=0)
            )

        elif shap_array.ndim == 3:
            mean_absolute_values = (
                np.abs(shap_array)
                .mean(axis=(0, 2))
            )

        else:
            raise ValueError(
                "Unexpected SHAP output "
                f"shape: {shap_array.shape}"
            )

    if len(
        mean_absolute_values
    ) != len(feature_names):
        raise ValueError(
            "SHAP feature count does not "
            "match transformed feature count. "
            f"SHAP features: "
            f"{len(mean_absolute_values)}, "
            f"Feature names: "
            f"{len(feature_names)}"
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": (
                mean_absolute_values
            ),
        }
    )

    total = result[
        "mean_absolute_shap"
    ].sum()

    if total > 0:
        result[
            "importance_percentage"
        ] = (
            result[
                "mean_absolute_shap"
            ]
            / total
            * 100
        )
    else:
        result[
            "importance_percentage"
        ] = 0.0

    result = result.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result[
        "rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    return result[
        [
            "rank",
            "feature",
            "mean_absolute_shap",
            "importance_percentage",
        ]
    ]


def calculate_high_risk_error_shap(
    model,
    transformed: np.ndarray,
    feature_names: list[str],
    actual_labels: np.ndarray,
    predictions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate SHAP importance for High_Risk classification errors.

    Parameters
    ----------
    model : object
        LightGBM estimator.

    transformed : np.ndarray
        Transformed test data.

    feature_names : list[str]
        Transformed feature names.

    actual_labels : np.ndarray
        Actual class labels.

    predictions : np.ndarray
        Predicted class labels.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        SHAP importance and summary dataframes.
    """
    mask = (
        (actual_labels == HIGH_RISK_CLASS)
        & (predictions != HIGH_RISK_CLASS)
    )

    error_indices = np.where(mask)[0]

    if len(error_indices) == 0:
        logger.info(
            "No High_Risk errors found."
        )

        importance = pd.DataFrame(
            columns=[
                "rank",
                "feature",
                "mean_absolute_shap",
                "importance_percentage",
            ]
        )

        summary = pd.DataFrame(
            {
                "metric": ["high_risk_errors"],
                "value": [0],
            }
        )

        return importance, summary

    logger.info(
        "Calculating SHAP for %d "
        "High_Risk errors.",
        len(error_indices),
    )

    error_transformed = transformed[
        error_indices
    ]

    importance = calculate_shap_importance(
        model,
        error_transformed,
        feature_names,
    )

    summary = pd.DataFrame(
        {
            "metric": ["high_risk_errors"],
            "value": [len(error_indices)],
        }
    )

    return importance, summary


def run_explainability() -> None:
    """
    Execute global SHAP explainability analysis.
    """
    logger.info(
        "Starting LightGBM classification "
        "explainability."
    )

    test_data = load_test_data()

    model_artifact = load_pickle_artifact(
        MODEL_PATH
    )

    preprocessor = load_pickle_artifact(
        PREPROCESSOR_PATH
    )
    
    label_mapping = load_pickle_artifact(
        LABEL_MAPPING_PATH
    )
    
    model = get_model_from_artifact(
        model_artifact
    )

    features = prepare_features(
        test_data
    )

    transformed = transform_features(
        preprocessor,
        features,
    )

    logger.info(
        "Transformed test shape: %s",
        transformed.shape,
    )

    feature_names = get_feature_names(
        preprocessor,
        transformed,
    )

    predictions, actual_labels = (
        get_predictions_and_class_labels(
            model,
            transformed,
            test_data,
            label_mapping,
        )
    )
    
    print("\nACTUAL LABEL DISTRIBUTION")
    print(pd.Series(actual_labels).value_counts().to_string())

    print("\nPREDICTED LABEL DISTRIBUTION")
    print(pd.Series(predictions).value_counts().to_string())

    print("\nCONFUSION PAIRS")
    print(
        pd.DataFrame(
            {
                "actual": actual_labels,
                "predicted": predictions,
            }
        )
        .value_counts()
        .to_string()
    )

    print("\nHIGH_RISK ERROR COUNT")
    print(
        (
            (actual_labels == HIGH_RISK_CLASS)
            & (predictions != HIGH_RISK_CLASS)
        ).sum()
    )

    high_risk_shap, high_risk_summary = (
        calculate_high_risk_error_shap(
            model,
            transformed,
            feature_names,
            actual_labels,
            predictions,
        )
    )
    
    
    logger.info(
        "Transformed feature count: %d",
        len(feature_names),
    )

    importance = calculate_shap_importance(
        model,
        transformed,
        feature_names,
    )

    EXPLAINABILITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    high_risk_shap.to_csv(
        HIGH_RISK_ERROR_SHAP_OUTPUT,
        index=False,
    )

    high_risk_summary.to_csv(
        HIGH_RISK_ERROR_SUMMARY_OUTPUT,
        index=False,
    )
    
    if not HIGH_RISK_ERROR_SHAP_OUTPUT.exists():
        raise RuntimeError(
            "High_Risk SHAP output file "
            "was not created."
        )

    if not HIGH_RISK_ERROR_SUMMARY_OUTPUT.exists():
        raise RuntimeError(
            "High_Risk SHAP summary file "
            "was not created."
        )
    
    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            "SHAP output file was not created."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "LIGHTGBM CLASSIFICATION "
        "EXPLAINABILITY"
    )

    print(
        "=" * 70
    )

    print(
        f"Test records: "
        f"{len(test_data):,}"
    )

    print(
        f"Transformed features: "
        f"{transformed.shape[1]}"
    )

    print(
        "\nTop 20 Features by Mean "
        "Absolute SHAP Value"
    )

    print(
        "-" * 70
    )

    print(
        importance.head(
            TOP_N
        ).to_string(
            index=False
        )
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nHigh_Risk Error Summary"
    )

    print(
        "-" * 70
    )

    print(
        high_risk_summary.to_string(
            index=False
        )
    )

    print(
        "\nHigh_Risk Error SHAP Outputs:"
    )

    print(
        HIGH_RISK_ERROR_SHAP_OUTPUT
    )

    print(
        HIGH_RISK_ERROR_SUMMARY_OUTPUT
    )
    
    logger.info(
        "LightGBM classification "
        "explainability completed."
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

    run_explainability()