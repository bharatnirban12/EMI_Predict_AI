"""
Controlled XGBoost hyperparameter tuning for EMI regression.

The existing XGBoost baseline is not modified.

Design:
    - preprocessing is fitted only on training data
    - validation data is used for hyperparameter selection
    - test data is evaluated only after model selection
    - baseline configuration is included in the search
    - all experiments are reproducible
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from src.features.feature_engineering import engineer_features
from src.preprocessing.preprocessing_pipeline import (
    create_preprocessing_pipeline,
    prepare_features_and_target,
)


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "modeling"

TRAIN_PATH = DATA_DIR / "regression_train.csv"
VALIDATION_PATH = DATA_DIR / "regression_validation.csv"
TEST_PATH = DATA_DIR / "regression_test.csv"

RESULTS_PATH = (
    REPORT_DIR / "xgboost_tuning_results.csv"
)

TARGET_COLUMN = "max_monthly_emi"

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Baseline configuration
# ---------------------------------------------------------------------

BASELINE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 1,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
}


# ---------------------------------------------------------------------
# Controlled search space
# ---------------------------------------------------------------------

PARAM_GRID = {
    "n_estimators": [
        200,
        300,
        400,
    ],
    "max_depth": [
        5,
        6,
        7,
    ],
    "learning_rate": [
        0.03,
        0.05,
        0.08,
    ],
    "min_child_weight": [
        1,
        3,
    ],
    "subsample": [
        0.80,
        1.00,
    ],
    "colsample_bytree": [
        0.80,
        1.00,
    ],
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a processed regression dataset."""
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


def prepare_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply feature engineering and separate target."""
    engineered = engineer_features(dataframe)

    features, target = prepare_features_and_target(
        engineered,
        TARGET_COLUMN,
    )

    if TARGET_COLUMN in features.columns:
        raise ValueError(
            "Target leakage detected."
        )

    return features, target


def validate_columns(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> None:
    """Validate predictor-column consistency."""
    train_columns = list(train_features.columns)

    if train_columns != list(validation_features.columns):
        raise ValueError(
            "Training and validation columns differ."
        )

    if train_columns != list(test_features.columns):
        raise ValueError(
            "Training and test columns differ."
        )


def calculate_metrics(
    model: XGBRegressor,
    features,
    target: pd.Series,
) -> dict[str, float]:
    """Calculate regression metrics."""
    predictions = model.predict(features)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Predictions contain NaN or infinite values."
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

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def build_parameter_combinations() -> list[dict]:
    """Create deterministic parameter combinations."""
    keys = list(PARAM_GRID.keys())

    combinations = []

    for values in itertools.product(
        *(PARAM_GRID[key] for key in keys)
    ):
        parameters = dict(
            zip(keys, values)
        )

        combinations.append(parameters)

    baseline_exists = any(
        parameters == BASELINE_PARAMS
        for parameters in combinations
    )

    if not baseline_exists:
        combinations.insert(
            0,
            BASELINE_PARAMS.copy(),
        )

    return combinations


def train_and_validate(
    parameters: dict,
    X_train,
    y_train: pd.Series,
    X_validation,
    y_validation: pd.Series,
    experiment_number: int,
    total_experiments: int,
) -> dict:
    """Train one XGBoost configuration."""
    logger.info(
        "Experiment %d/%d: %s",
        experiment_number,
        total_experiments,
        parameters,
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=parameters["n_estimators"],
        max_depth=parameters["max_depth"],
        learning_rate=parameters["learning_rate"],
        min_child_weight=parameters["min_child_weight"],
        subsample=parameters["subsample"],
        colsample_bytree=parameters[
            "colsample_bytree"
        ],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    metrics = calculate_metrics(
        model,
        X_validation,
        y_validation,
    )

    return {
        **parameters,
        "validation_mae": metrics["mae"],
        "validation_rmse": metrics["rmse"],
        "validation_r2": metrics["r2"],
    }


def main() -> None:
    """Run controlled XGBoost hyperparameter tuning."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    logger.info(
        "Starting controlled XGBoost tuning."
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

    X_train, y_train = prepare_dataset(
        train_data
    )

    X_validation, y_validation = prepare_dataset(
        validation_data
    )

    X_test, y_test = prepare_dataset(
        test_data
    )

    validate_columns(
        X_train,
        X_validation,
        X_test,
    )

    logger.info(
        "Training predictors: %s",
        X_train.shape,
    )

    logger.info(
        "Validation predictors: %s",
        X_validation.shape,
    )

    logger.info(
        "Test predictors: %s",
        X_test.shape,
    )

    # Fit preprocessing ONLY on training data.
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

    if not (
        X_train_transformed.shape[1]
        == X_validation_transformed.shape[1]
        == X_test_transformed.shape[1]
    ):
        raise ValueError(
            "Transformed feature counts differ."
        )

    parameter_combinations = (
        build_parameter_combinations()
    )

    total_experiments = len(
        parameter_combinations
    )

    logger.info(
        "Total tuning configurations: %d",
        total_experiments,
    )

    results = []

    for index, parameters in enumerate(
        parameter_combinations,
        start=1,
    ):
        result = train_and_validate(
            parameters=parameters,
            X_train=X_train_transformed,
            y_train=y_train,
            X_validation=X_validation_transformed,
            y_validation=y_validation,
            experiment_number=index,
            total_experiments=total_experiments,
        )

        results.append(result)

        logger.info(
            "Validation RMSE: %.4f | "
            "MAE: %.4f | R2: %.4f",
            result["validation_rmse"],
            result["validation_mae"],
            result["validation_r2"],
        )

    results_dataframe = pd.DataFrame(
        results
    )

    results_dataframe = (
        results_dataframe.sort_values(
            by=[
                "validation_rmse",
                "validation_mae",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    results_dataframe.insert(
        0,
        "rank",
        range(
            1,
            len(results_dataframe) + 1,
        ),
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        RESULTS_PATH,
        index=False,
    )

    best_row = results_dataframe.iloc[0]

    best_parameters = {
        parameter: best_row[parameter]
        for parameter in PARAM_GRID
    }

    logger.info(
        "Best validation configuration: %s",
        best_parameters,
    )

    # -------------------------------------------------------------
    # Final model fit using selected parameters.
    #
    # IMPORTANT:
    # The test set has not influenced parameter selection.
    # -------------------------------------------------------------

    final_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(
            best_parameters["n_estimators"]
        ),
        max_depth=int(
            best_parameters["max_depth"]
        ),
        learning_rate=float(
            best_parameters["learning_rate"]
        ),
        min_child_weight=int(
            best_parameters["min_child_weight"]
        ),
        subsample=float(
            best_parameters["subsample"]
        ),
        colsample_bytree=float(
            best_parameters["colsample_bytree"]
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info(
        "Training selected XGBoost configuration."
    )

    final_model.fit(
        X_train_transformed,
        y_train,
    )

    validation_metrics = calculate_metrics(
        final_model,
        X_validation_transformed,
        y_validation,
    )

    test_metrics = calculate_metrics(
        final_model,
        X_test_transformed,
        y_test,
    )

    print()
    print("=" * 90)
    print("XGBOOST HYPERPARAMETER TUNING SUMMARY")
    print("=" * 90)

    print(
        f"Configurations evaluated: "
        f"{total_experiments}"
    )

    print(
        "\nBest Parameters:"
    )

    for parameter, value in best_parameters.items():
        print(
            f"{parameter}: {value}"
        )

    print(
        "\nBest Validation Metrics:"
    )

    print(
        f"MAE:  {validation_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {validation_metrics['rmse']:.4f}"
    )

    print(
        f"R2:   {validation_metrics['r2']:.4f}"
    )

    print(
        "\nFinal Test Metrics:"
    )

    print(
        f"MAE:  {test_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: {test_metrics['rmse']:.4f}"
    )

    print(
        f"R2:   {test_metrics['r2']:.4f}"
    )

    print(
        "\nTop configurations:"
    )

    print(
        results_dataframe.head(10).to_string(
            index=False
        )
    )

    print(
        "\nResults saved:"
    )

    print(
        RESULTS_PATH
    )

    logger.info(
        "XGBoost tuning completed successfully."
    )


if __name__ == "__main__":
    main()