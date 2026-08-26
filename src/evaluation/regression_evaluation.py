"""
Module: regression_evaluation.py

Description:
    Evaluate the selected XGBoost regression model on the
    held-out regression test dataset.

    This module performs:
        - Test-set prediction
        - MAE calculation
        - RMSE calculation
        - R² calculation
        - Error analysis
        - Feature importance extraction
        - Requirement validation

    The saved preprocessing artifact from the XGBoost training
    stage is reused without refitting.

Author:
    EMIPredict AI Project
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.features.feature_engineering import (
    engineer_features,
)
from src.preprocessing.preprocessing_pipeline import (
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

EVALUATION_DIR = (
    ARTIFACTS_DIR
    / "evaluation"
)

TEST_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "regression_test.csv"
)

MODEL_PATH = (
    ARTIFACTS_DIR
    / "xgboost_regressor_model.pkl"
)

PREPROCESSOR_PATH = (
    ARTIFACTS_DIR
    / "xgboost_regressor_preprocessor.pkl"
)

ERROR_ANALYSIS_PATH = (
    EVALUATION_DIR
    / "xgboost_error_analysis.csv"
)

FEATURE_IMPORTANCE_PATH = (
    EVALUATION_DIR
    / "xgboost_feature_importance.csv"
)

METRICS_PATH = (
    EVALUATION_DIR
    / "xgboost_test_metrics.csv"
)

TARGET_COLUMN = "max_monthly_emi"

MAX_ERROR_RECORDS = 100


def load_pickle_artifact(
    file_path: Path,
):
    """
    Load a serialized artifact.

    Parameters
    ----------
    file_path : Path
        Artifact path.

    Returns
    -------
    object
        Loaded artifact.
    """
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


def load_test_dataset() -> pd.DataFrame:
    """
    Load the regression test dataset.

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
            "Regression test dataset is empty."
        )

    logger.info(
        "Loaded regression test dataset: %s",
        dataframe.shape,
    )

    return dataframe


def prepare_test_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply the project's feature engineering and separate
    predictors from the regression target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw regression test split.

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
            f"'{TARGET_COLUMN}' is present "
            "in predictors."
        )

    logger.info(
        "Prepared test predictors: %s",
        features.shape,
    )

    logger.info(
        "Prepared test target: %s",
        target.shape,
    )

    return features, target


def calculate_metrics(
    target: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Parameters
    ----------
    target : pd.Series
        Actual target values.

    predictions : np.ndarray
        Model predictions.

    Returns
    -------
    dict[str, float]
        MAE, RMSE, and R².
    """
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

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def create_error_analysis(
    original_data: pd.DataFrame,
    target: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Create a prediction error analysis dataframe.

    Parameters
    ----------
    original_data : pd.DataFrame
        Original regression test records.

    target : pd.Series
        Actual target values.

    predictions : np.ndarray
        Model predictions.

    Returns
    -------
    pd.DataFrame
        Error analysis dataframe.
    """
    error_analysis = original_data.copy()

    error_analysis[
        "actual_max_monthly_emi"
    ] = target.to_numpy()

    error_analysis[
        "predicted_max_monthly_emi"
    ] = predictions

    error_analysis[
        "absolute_error"
    ] = np.abs(
        error_analysis[
            "actual_max_monthly_emi"
        ]
        - error_analysis[
            "predicted_max_monthly_emi"
        ]
    )

    error_analysis[
        "signed_error"
    ] = (
        error_analysis[
            "predicted_max_monthly_emi"
        ]
        - error_analysis[
            "actual_max_monthly_emi"
        ]
    )

    error_analysis = error_analysis.sort_values(
        by="absolute_error",
        ascending=False,
    )

    return error_analysis


def extract_feature_importance(
    model,
    preprocessor,
) -> pd.DataFrame:
    """
    Extract model feature importance values.

    Parameters
    ----------
    model
        Fitted XGBoost model.

    preprocessor
        Fitted preprocessing pipeline.

    Returns
    -------
    pd.DataFrame
        Feature importance table.
    """
    if not hasattr(
        model,
        "feature_importances_",
    ):
        raise AttributeError(
            "Selected model does not expose "
            "feature_importances_."
        )

    importance_values = (
        model.feature_importances_
    )

    try:
        feature_names = (
            preprocessor.get_feature_names_out()
        )
    except AttributeError as exc:
        raise AttributeError(
            "Preprocessor does not expose "
            "get_feature_names_out()."
        ) from exc

    if len(importance_values) != len(
        feature_names
    ):
        raise ValueError(
            "Feature importance count does not "
            "match transformed feature count."
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    importance = importance.reset_index(
        drop=True
    )

    return importance


def validate_requirements(
    metrics: dict[str, float],
) -> None:
    """
    Validate the project regression requirements.

    Parameters
    ----------
    metrics : dict[str, float]
        Test metrics.

    Raises
    ------
    ValueError
        If the model fails a required threshold.
    """
    mae_pass = metrics["mae"] < 500
    r2_pass = metrics["r2"] > 0.85

    logger.info(
        "Regression MAE requirement: %s",
        "PASS" if mae_pass else "FAIL",
    )

    logger.info(
        "Regression R² requirement: %s",
        "PASS" if r2_pass else "FAIL",
    )

    if not mae_pass or not r2_pass:
        raise ValueError(
            "Selected regression model does not "
            "satisfy all project requirements."
        )


def save_dataframe(
    dataframe: pd.DataFrame,
    file_path: Path,
) -> None:
    """
    Save a dataframe as CSV.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data to save.

    file_path : Path
        Output path.
    """
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        file_path,
        index=False,
    )

    if not file_path.exists():
        raise RuntimeError(
            f"Output file was not created: "
            f"{file_path}"
        )

    logger.info(
        "Saved evaluation output: %s",
        file_path,
    )


def evaluate_regression_model() -> None:
    """
    Execute final regression model evaluation.
    """
    logger.info(
        "Starting final XGBoost regression evaluation."
    )

    test_data = load_test_dataset()

    model = load_pickle_artifact(
        MODEL_PATH
    )

    preprocessor = load_pickle_artifact(
        PREPROCESSOR_PATH
    )

    features, target = prepare_test_data(
        test_data
    )

    transformed_features = (
        preprocessor.transform(
            features
        )
    )

    logger.info(
        "Transformed test shape: %s",
        transformed_features.shape,
    )

    predictions = model.predict(
        transformed_features
    )

    predictions = np.asarray(
        predictions
    )

    if predictions.shape[0] != target.shape[0]:
        raise ValueError(
            "Prediction count does not match "
            "test target count."
        )

    if not np.isfinite(
        predictions
    ).all():
        raise ValueError(
            "Model predictions contain "
            "NaN or infinite values."
        )

    metrics = calculate_metrics(
        target,
        predictions,
    )

    validate_requirements(
        metrics
    )

    error_analysis = create_error_analysis(
        test_data,
        target,
        predictions,
    )

    top_errors = error_analysis.head(
        MAX_ERROR_RECORDS
    )

    feature_importance = (
        extract_feature_importance(
            model,
            preprocessor,
        )
    )

    metrics_dataframe = pd.DataFrame(
        [
            {
                "model": "XGBoost Regressor",
                "dataset": "test",
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mae_requirement_pass": (
                    metrics["mae"] < 500
                ),
                "r2_requirement_pass": (
                    metrics["r2"] > 0.85
                ),
            }
        ]
    )

    save_dataframe(
        top_errors,
        ERROR_ANALYSIS_PATH,
    )

    save_dataframe(
        feature_importance,
        FEATURE_IMPORTANCE_PATH,
    )

    save_dataframe(
        metrics_dataframe,
        METRICS_PATH,
    )

    print(
        "\n" + "=" * 60
    )
    print(
        "FINAL XGBOOST REGRESSION EVALUATION"
    )
    print(
        "=" * 60
    )

    print(
        f"Test records: {len(target):,}"
    )

    print(
        f"Transformed features: "
        f"{transformed_features.shape[1]}"
    )

    print(
        "\nTest Metrics:"
    )

    print(
        f"MAE:  {metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {metrics['rmse']:.4f}"
    )

    print(
        f"R²:   {metrics['r2']:.4f}"
    )

    print(
        "\nRequirement Validation:"
    )

    print(
        "MAE < 500: "
        f"{'PASS' if metrics['mae'] < 500 else 'FAIL'}"
    )

    print(
        "R² > 0.85: "
        f"{'PASS' if metrics['r2'] > 0.85 else 'FAIL'}"
    )

    print(
        "\nTop 10 Features:"
    )

    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )

    print(
        "\nLargest Prediction Errors:"
    )

    error_columns = [
        "actual_max_monthly_emi",
        "predicted_max_monthly_emi",
        "absolute_error",
        "signed_error",
    ]

    print(
        top_errors[
            error_columns
        ].head(10).to_string(
            index=False
        )
    )

    print(
        "\nEvaluation Outputs:"
    )

    print(
        f"Error analysis: "
        f"{ERROR_ANALYSIS_PATH}"
    )

    print(
        f"Feature importance: "
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    print(
        f"Metrics: "
        f"{METRICS_PATH}"
    )

    logger.info(
        "Final XGBoost regression evaluation completed."
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

    evaluate_regression_model()