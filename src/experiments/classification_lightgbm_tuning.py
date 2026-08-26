"""
LightGBM hyperparameter tuning for EMI eligibility classification.

The experiment:
1. Loads the predefined train/validation/test datasets.
2. Applies the existing feature-engineering pipeline.
3. Fits preprocessing on training data only.
4. Evaluates multiple LightGBM configurations on validation data.
5. Selects the best configuration using validation macro-F1.
6. Evaluates the selected configuration on the untouched test set.
7. Saves all tuning results to reports/modeling.

The test set is never used to select hyperparameters.
"""

from __future__ import annotations

import logging
import pickle
from itertools import product
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "modeling"
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

RESULTS_PATH = (
    REPORTS_DIR
    / "lightgbm_classification_tuning_results.csv"
)

TARGET_COLUMN = "emi_eligibility"
RANDOM_STATE = 42

PARAM_GRID = {
    "n_estimators": [200, 300, 400],
    "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 6, 10],
    "min_child_samples": [20, 50],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a classification dataset.

    Parameters
    ----------
    file_path:
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
    Apply existing feature engineering and separate target.

    Parameters
    ----------
    dataframe:
        Raw processed classification dataframe.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Predictors and target.
    """
    engineered_data = engineer_features(
        dataframe
    )

    features, target = prepare_features_and_target(
        engineered_data,
        TARGET_COLUMN,
    )

    return features, target


def encode_targets(
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
    Encode target labels using the training classes only.

    Parameters
    ----------
    train_target:
        Training target.

    validation_target:
        Validation target.

    test_target:
        Test target.

    Returns
    -------
    tuple
        Encoded targets and label mapping.

    Raises
    ------
    ValueError
        If validation/test contains an unseen class.
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
            "Validation contains classes absent "
            f"from training data: "
            f"{sorted(unknown_validation)}"
        )

    if unknown_test:
        raise ValueError(
            "Test contains classes absent "
            f"from training data: "
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

    return (
        train_encoded,
        validation_encoded,
        test_encoded,
        label_mapping,
    )


def calculate_metrics(
    y_true: pd.Series,
    predictions,
    label_mapping: dict[str, int],
) -> dict[str, float]:
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true:
        Encoded target.

    predictions:
        Model predictions.

    label_mapping:
        Class-to-integer mapping.

    Returns
    -------
    dict[str, float]
        Classification metrics.
    """
    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    weighted_precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    high_risk_id = label_mapping.get(
        "High_Risk"
    )

    if high_risk_id is None:
        raise ValueError(
            "'High_Risk' class was not found "
            "in the target mapping."
        )

    high_risk_precision = precision_score(
        y_true,
        predictions,
        labels=[high_risk_id],
        average="macro",
        zero_division=0,
    )

    high_risk_recall = recall_score(
        y_true,
        predictions,
        labels=[high_risk_id],
        average="macro",
        zero_division=0,
    )

    high_risk_f1 = f1_score(
        y_true,
        predictions,
        labels=[high_risk_id],
        average="macro",
        zero_division=0,
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
        "high_risk_precision": float(
            high_risk_precision
        ),
        "high_risk_recall": float(
            high_risk_recall
        ),
        "high_risk_f1": float(
            high_risk_f1
        ),
    }


def generate_configurations() -> list[dict]:
    """
    Generate the Cartesian product of the parameter grid.

    Returns
    -------
    list[dict]
        List of LightGBM configurations.
    """
    parameter_names = list(
        PARAM_GRID.keys()
    )

    parameter_values = [
        PARAM_GRID[name]
        for name in parameter_names
    ]

    configurations = []

    for values in product(
        *parameter_values
    ):
        configuration = dict(
            zip(
                parameter_names,
                values,
            )
        )

        configurations.append(
            configuration
        )

    return configurations


def train_and_evaluate_configuration(
    configuration: dict,
    X_train,
    y_train,
    X_validation,
    y_validation,
    label_mapping: dict[str, int],
    configuration_number: int,
    total_configurations: int,
) -> dict[str, float]:
    """
    Train one LightGBM configuration and evaluate it.

    Test data is deliberately not passed to this function.

    Parameters
    ----------
    configuration:
        LightGBM hyperparameters.

    X_train:
        Transformed training features.

    y_train:
        Encoded training target.

    X_validation:
        Transformed validation features.

    y_validation:
        Encoded validation target.

    label_mapping:
        Target label mapping.

    configuration_number:
        Current configuration number.

    total_configurations:
        Total number of configurations.

    Returns
    -------
    dict[str, float]
        Configuration and validation metrics.
    """
    logger.info(
        "Configuration %d/%d: %s",
        configuration_number,
        total_configurations,
        configuration,
    )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(label_mapping),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **configuration,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    metrics = calculate_metrics(
        y_validation,
        predictions,
        label_mapping,
    )

    result = {
        "configuration_number": configuration_number,
        **configuration,
        "validation_accuracy": metrics[
            "accuracy"
        ],
        "validation_weighted_precision": metrics[
            "weighted_precision"
        ],
        "validation_weighted_recall": metrics[
            "weighted_recall"
        ],
        "validation_weighted_f1": metrics[
            "weighted_f1"
        ],
        "validation_macro_f1": metrics[
            "macro_f1"
        ],
        "validation_high_risk_precision": metrics[
            "high_risk_precision"
        ],
        "validation_high_risk_recall": metrics[
            "high_risk_recall"
        ],
        "validation_high_risk_f1": metrics[
            "high_risk_f1"
        ],
    }

    logger.info(
        "Configuration %d/%d results: "
        "macro_f1=%.4f, high_risk_f1=%.4f, "
        "high_risk_recall=%.4f, accuracy=%.4f",
        configuration_number,
        total_configurations,
        metrics["macro_f1"],
        metrics["high_risk_f1"],
        metrics["high_risk_recall"],
        metrics["accuracy"],
    )

    return result


def save_results(
    results: list[dict],
) -> pd.DataFrame:
    """
    Save tuning results to CSV.

    Parameters
    ----------
    results:
        Tuning result records.

    Returns
    -------
    pd.DataFrame
        Sorted results dataframe.
    """
    results_dataframe = pd.DataFrame(
        results
    )

    results_dataframe = (
        results_dataframe.sort_values(
            by=[
                "validation_macro_f1",
                "validation_high_risk_f1",
                "validation_high_risk_recall",
                "validation_weighted_f1",
            ],
            ascending=False,
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

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        RESULTS_PATH,
        index=False,
    )

    logger.info(
        "Saved tuning results: %s",
        RESULTS_PATH,
    )

    return results_dataframe


def evaluate_best_model_on_test(
    best_configuration: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    label_mapping: dict[str, int],
) -> dict[str, float]:
    """
    Train the selected configuration on training data and
    evaluate it on the untouched test data.

    The test set is used only after hyperparameter selection.

    Parameters
    ----------
    best_configuration:
        Selected hyperparameters.

    X_train:
        Transformed training features.

    y_train:
        Encoded training target.

    X_test:
        Transformed test features.

    y_test:
        Encoded test target.

    label_mapping:
        Target label mapping.

    Returns
    -------
    dict[str, float]
        Final test metrics.
    """
    logger.info(
        "Training best LightGBM configuration "
        "for final test evaluation."
    )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(label_mapping),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **best_configuration,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    return calculate_metrics(
        y_test,
        predictions,
        label_mapping,
    )


def main() -> None:
    """
    Execute the LightGBM hyperparameter tuning experiment.
    """
    logger.info(
        "Starting LightGBM classification tuning."
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

    preprocessor = (
        create_preprocessing_pipeline(
            X_train
        )
    )

    logger.info(
        "Fitting preprocessing pipeline "
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

    (
        y_train_encoded,
        y_validation_encoded,
        y_test_encoded,
        label_mapping,
    ) = encode_targets(
        y_train,
        y_validation,
        y_test,
    )

    logger.info(
        "Target mapping: %s",
        label_mapping,
    )

    configurations = (
        generate_configurations()
    )

    total_configurations = len(
        configurations
    )

    logger.info(
        "Total LightGBM configurations: %d",
        total_configurations,
    )

    results = []

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):
        result = (
            train_and_evaluate_configuration(
                configuration=configuration,
                X_train=X_train_transformed,
                y_train=y_train_encoded,
                X_validation=X_validation_transformed,
                y_validation=y_validation_encoded,
                label_mapping=label_mapping,
                configuration_number=index,
                total_configurations=(
                    total_configurations
                ),
            )
        )

        results.append(result)

    results_dataframe = save_results(
        results
    )

    best_row = results_dataframe.iloc[0]

    hyperparameter_names = list(
        PARAM_GRID.keys()
    )

    best_configuration = {
        parameter: best_row[parameter]
        for parameter in hyperparameter_names
    }

    for parameter in (
        "n_estimators",
        "num_leaves",
        "max_depth",
        "min_child_samples",
    ):
        best_configuration[parameter] = int(
            best_configuration[parameter]
        )

    logger.info(
        "Best validation configuration: %s",
        best_configuration,
    )

    test_metrics = (
        evaluate_best_model_on_test(
            best_configuration=best_configuration,
            X_train=X_train_transformed,
            y_train=y_train_encoded,
            X_test=X_test_transformed,
            y_test=y_test_encoded,
            label_mapping=label_mapping,
        )
    )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "LIGHTGBM CLASSIFICATION HYPERPARAMETER TUNING"
    )
    print(
        "=" * 80
    )

    print(
        f"Configurations evaluated: "
        f"{total_configurations}"
    )

    print(
        "\nBest Parameters:"
    )

    for parameter in hyperparameter_names:
        print(
            f"{parameter}: "
            f"{best_configuration[parameter]}"
        )

    print(
        "\nBest Validation Metrics:"
    )

    print(
        f"Accuracy: "
        f"{best_row['validation_accuracy']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{best_row['validation_weighted_f1']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{best_row['validation_macro_f1']:.4f}"
    )

    print(
        f"High_Risk Precision: "
        f"{best_row['validation_high_risk_precision']:.4f}"
    )

    print(
        f"High_Risk Recall: "
        f"{best_row['validation_high_risk_recall']:.4f}"
    )

    print(
        f"High_Risk F1: "
        f"{best_row['validation_high_risk_f1']:.4f}"
    )

    print(
        "\nFinal Test Metrics:"
    )

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"High_Risk Precision: "
        f"{test_metrics['high_risk_precision']:.4f}"
    )

    print(
        f"High_Risk Recall: "
        f"{test_metrics['high_risk_recall']:.4f}"
    )

    print(
        f"High_Risk F1: "
        f"{test_metrics['high_risk_f1']:.4f}"
    )

    print(
        "\nResults saved:"
    )

    print(
        RESULTS_PATH
    )

    print(
        "=" * 80
    )

    logger.info(
        "LightGBM classification tuning completed."
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