"""
Module: regression_explainability.py

Description:
    Generate global SHAP-based feature importance for the
    selected XGBoost regression model.

    The existing trained model and preprocessing artifact are
    reused. No model retraining is performed.

    A JSON representation of the underlying XGBoost Booster is
    used for SHAP compatibility with XGBoost 3.2.0 and SHAP 0.49.1.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "SHAP is required for regression explainability. "
        "Install it with: pip install shap"
    ) from exc

from src.features.feature_engineering import (
    engineer_features,
)
from src.features.preprocessing_pipeline import (
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

REGRESSION_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "regression"
)

EXPLAINABILITY_DIR = (
    REGRESSION_ARTIFACT_DIR
    / "explainability"
)

TEST_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "regression_test.csv"
)

MODEL_PATH = (
    REGRESSION_ARTIFACT_DIR
    / "xgboost_regressor_model.pkl"
)

PREPROCESSOR_PATH = (
    REGRESSION_ARTIFACT_DIR
    / "xgboost_regressor_preprocessor.pkl"
)

COMPATIBILITY_MODEL_PATH = (
    EXPLAINABILITY_DIR
    / "xgboost_shap_compat.json"
)

OUTPUT_PATH = (
    EXPLAINABILITY_DIR
    / "xgboost_shap_feature_importance.csv"
)

TARGET_COLUMN = "max_monthly_emi"

TOP_N = 30


def load_pickle_artifact(path: Path):
    """
    Load a serialized artifact.

    Parameters
    ----------
    path : Path
        Path to serialized artifact.

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
    Load the regression test dataset.

    Returns
    -------
    pd.DataFrame
        Regression test dataset.
    """
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Regression test dataset not found: "
            f"{TEST_DATA_PATH}"
        )

    dataframe = pd.read_csv(
        TEST_DATA_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "Regression test dataset is empty."
        )

    logger.info(
        "Loaded regression test dataset: %s",
        dataframe.shape,
    )

    return dataframe


def prepare_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply feature engineering and prepare regression predictors.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Regression test dataset.

    Returns
    -------
    pd.DataFrame
        Predictor dataframe.
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
            f"{TARGET_COLUMN} exists in predictors."
        )

    if target is None:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not returned."
        )

    logger.info(
        "Prepared predictors: %s",
        features.shape,
    )

    logger.info(
        "Prepared target: %s",
        target.shape,
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

    Returns
    -------
    list[str]
        Transformed feature names.
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

    logger.warning(
        "Transformed feature names were not "
        "available from the preprocessor. "
        "Generating fallback names."
    )

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
    Extract the regression estimator from a serialized artifact.

    Parameters
    ----------
    model_artifact : object
        Loaded model artifact.

    Returns
    -------
    object
        Regression estimator.
    """
    if isinstance(
        model_artifact,
        dict,
    ):
        for key in (
            "model",
            "estimator",
            "regressor",
        ):
            if key in model_artifact:
                return model_artifact[key]

    return model_artifact



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
        XGBRegressor model.

    transformed : np.ndarray
        Transformed test data.

    feature_names : list[str]
        Transformed feature names.

    Returns
    -------
    pd.DataFrame
        SHAP feature importance dataframe.
    """
    EXPLAINABILITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not hasattr(model, "get_booster"):
        raise TypeError(
            "Expected an XGBRegressor exposing "
            "get_booster(). Received: "
            f"{type(model)}"
        )

    logger.info(
        "Saving SHAP-compatible XGBoost Booster JSON."
    )

    booster = model.get_booster()

    booster.save_model(
        str(COMPATIBILITY_MODEL_PATH)
    )

    if not COMPATIBILITY_MODEL_PATH.exists():
        raise RuntimeError(
            "Failed to create SHAP-compatible "
            "XGBoost JSON model."
        )

    logger.info(
        "SHAP-compatible Booster saved: %s",
        COMPATIBILITY_MODEL_PATH,
    )

    compatible_booster = xgb.Booster()

    compatible_booster.load_model(
        str(COMPATIBILITY_MODEL_PATH)
    )

    logger.info(
        "SHAP-compatible Booster loaded as native "
        "xgboost.core.Booster."
    )

    logger.info(
        "Compatible Booster type: %s",
        type(compatible_booster),
    )

    logger.info(
        "Creating SHAP TreeExplainer from native Booster."
    )

    explainer = shap.TreeExplainer(
        compatible_booster
    )

    logger.info(
        "Calculating SHAP values."
    )

    shap_values = explainer.shap_values(
        transformed
    )

    shap_array = np.asarray(
        shap_values
    )

    if shap_array.ndim == 2:
        if shap_array.shape[1] != len(
            feature_names
        ):
            raise ValueError(
                "SHAP feature count does not "
                "match transformed feature count. "
                f"SHAP shape: {shap_array.shape}, "
                f"Feature names: "
                f"{len(feature_names)}"
            )

        mean_absolute_values = (
            np.abs(shap_array)
            .mean(axis=0)
        )

    elif shap_array.ndim == 1:
        if shap_array.shape[0] != len(
            feature_names
        ):
            raise ValueError(
                "SHAP feature count does not "
                "match transformed feature count. "
                f"SHAP shape: {shap_array.shape}, "
                f"Feature names: "
                f"{len(feature_names)}"
            )

        mean_absolute_values = np.abs(
            shap_array
        )

    else:
        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_array.shape}"
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


def run_explainability() -> None:
    """
    Execute global SHAP explainability analysis.
    """
    logger.info(
        "Starting XGBoost regression "
        "explainability."
    )

    test_data = load_test_data()

    model_artifact = load_pickle_artifact(
        MODEL_PATH
    )

    preprocessor = load_pickle_artifact(
        PREPROCESSOR_PATH
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

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            "SHAP output file was not created."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "XGBOOST REGRESSION "
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
        f"Target: {TARGET_COLUMN}"
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

    logger.info(
        "XGBoost regression "
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