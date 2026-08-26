"""
Module: split_data.py

Description:
    Create reproducible train, validation, and test datasets for
    the EMIPredict AI classification and regression tasks.

    Records identified as violating documented domain constraints
    are excluded from the model-development population but remain
    preserved in the domain-violation audit dataset.

Author:
    EMIPredict AI Project
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "cleaned_data.csv"
)

DOMAIN_VIOLATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "domain_violations.csv"
)

OUTPUT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CLASSIFICATION_TARGET = "emi_eligibility"
REGRESSION_TARGET = "max_monthly_emi"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def validate_split_configuration() -> None:
    """
    Validate train, validation, and test proportions.

    Raises
    ------
    ValueError
        If the split proportions do not sum to 1.
    """
    total = (
        TRAIN_SIZE
        + VALIDATION_SIZE
        + TEST_SIZE
    )

    if not abs(total - 1.0) < 1e-9:
        raise ValueError(
            "TRAIN_SIZE, VALIDATION_SIZE, and "
            "TEST_SIZE must sum to 1.0."
        )


def load_cleaned_dataset(
    file_path: Path = INPUT_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the cleaned dataset.

    Parameters
    ----------
    file_path : Path
        Path to the cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.

    Raises
    ------
    FileNotFoundError
        If the cleaned dataset does not exist.

    ValueError
        If the dataset is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            "The cleaned dataset is empty."
        )

    logger.info(
        "Loaded cleaned dataset: %s",
        dataframe.shape,
    )

    return dataframe


def load_domain_violation_report(
    file_path: Path = DOMAIN_VIOLATIONS_PATH,
) -> pd.DataFrame:
    """
    Load the domain-violation audit report.

    Parameters
    ----------
    file_path : Path
        Path to the domain-violation CSV.

    Returns
    -------
    pd.DataFrame
        Domain-violation report indexed by the original row index.

    Raises
    ------
    FileNotFoundError
        If the report does not exist.

    ValueError
        If required columns are missing or the report is empty.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Domain-violation report not found: {file_path}"
        )

    violations = pd.read_csv(
        file_path,
        index_col="original_row_index",
    )

    if violations.empty:
        raise ValueError(
            "The domain-violation report is empty."
        )

    required_columns = {
        "any_domain_violation",
    }

    missing_columns = (
        required_columns
        - set(violations.columns)
    )

    if missing_columns:
        raise ValueError(
            "Domain-violation report is missing "
            f"required columns: {sorted(missing_columns)}"
        )

    logger.info(
        "Loaded domain-violation report: %s",
        violations.shape,
    )

    return violations


def validate_targets(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that both required target columns exist.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to validate.

    Raises
    ------
    ValueError
        If either target is missing.
    """
    required_targets = {
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET,
    }

    missing_targets = [
        target
        for target in required_targets
        if target not in dataframe.columns
    ]

    if missing_targets:
        raise ValueError(
            "Missing required target columns: "
            f"{missing_targets}"
        )


def create_modeling_dataset(
    dataframe: pd.DataFrame,
    violations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove domain-invalid records from the modeling population.

    The original cleaned dataset is not modified.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset.

    violations : pd.DataFrame
        Domain-violation audit report.

    Returns
    -------
    pd.DataFrame
        Dataset containing only records without domain violations.

    Raises
    ------
    ValueError
        If the violation report does not align with the dataset.
    """
    if not dataframe.index.is_unique:
        raise ValueError(
            "Cleaned dataset index must be unique."
        )

    violation_indices = violations.index[
        violations["any_domain_violation"].astype(bool)
    ]

    missing_indices = violation_indices.difference(
        dataframe.index
    )

    if not missing_indices.empty:
        raise ValueError(
            "Domain-violation report contains row indices "
            "that are not present in the cleaned dataset. "
            f"Count: {len(missing_indices)}"
        )

    modeling_data = dataframe.drop(
        index=violation_indices,
    ).copy()

    excluded_count = (
        len(dataframe)
        - len(modeling_data)
    )

    logger.info(
        "Domain-invalid records excluded "
        "from model development: %d",
        excluded_count,
    )

    logger.info(
        "Model-development records retained: %d",
        len(modeling_data),
    )

    if modeling_data.empty:
        raise ValueError(
            "No records remain after domain-violation filtering."
        )

    return modeling_data


def split_classification_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train, validation, and test datasets
    for classification.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model-development dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation, and test datasets.
    """
    train_data, temporary_data = train_test_split(
        dataframe,
        test_size=(
            VALIDATION_SIZE
            + TEST_SIZE
        ),
        random_state=RANDOM_STATE,
        stratify=dataframe[CLASSIFICATION_TARGET],
    )

    validation_relative_size = (
        VALIDATION_SIZE
        / (VALIDATION_SIZE + TEST_SIZE)
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=1 - validation_relative_size,
        random_state=RANDOM_STATE,
        stratify=temporary_data[
            CLASSIFICATION_TARGET
        ],
    )

    logger.info(
        "Classification train shape: %s",
        train_data.shape,
    )

    logger.info(
        "Classification validation shape: %s",
        validation_data.shape,
    )

    logger.info(
        "Classification test shape: %s",
        test_data.shape,
    )

    return (
        train_data,
        validation_data,
        test_data,
    )


def split_regression_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train, validation, and test datasets
    for regression.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model-development dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation, and test datasets.
    """
    train_data, temporary_data = train_test_split(
        dataframe,
        test_size=(
            VALIDATION_SIZE
            + TEST_SIZE
        ),
        random_state=RANDOM_STATE,
    )

    validation_relative_size = (
        VALIDATION_SIZE
        / (VALIDATION_SIZE + TEST_SIZE)
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=1 - validation_relative_size,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Regression train shape: %s",
        train_data.shape,
    )

    logger.info(
        "Regression validation shape: %s",
        validation_data.shape,
    )

    logger.info(
        "Regression test shape: %s",
        test_data.shape,
    )

    return (
        train_data,
        validation_data,
        test_data,
    )


def save_split(
    dataframe: pd.DataFrame,
    file_path: Path,
) -> None:
    """
    Save a dataset split to CSV.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset split.

    file_path : Path
        Destination path.
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
            f"Failed to create file: {file_path}"
        )

    logger.info(
        "Saved split: %s",
        file_path,
    )


def log_class_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Log classification target distribution.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset split.

    dataset_name : str
        Name of the split.
    """
    distribution = (
        dataframe[CLASSIFICATION_TARGET]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    logger.info(
        "%s classification distribution:\n%s",
        dataset_name,
        distribution.to_string(),
    )


def create_data_splits() -> dict[str, Path]:
    """
    Create and save all classification and regression splits.

    Returns
    -------
    dict[str, Path]
        Mapping of split names to generated file paths.
    """
    validate_split_configuration()

    dataframe = load_cleaned_dataset()

    validate_targets(dataframe)

    violations = load_domain_violation_report()

    modeling_data = create_modeling_dataset(
        dataframe,
        violations,
    )

    logger.info(
        "Creating classification splits."
    )

    (
        classification_train,
        classification_validation,
        classification_test,
    ) = split_classification_data(
        modeling_data
    )

    logger.info(
        "Creating regression splits."
    )

    (
        regression_train,
        regression_validation,
        regression_test,
    ) = split_regression_data(
        modeling_data
    )

    paths = {
        "classification_train": (
            OUTPUT_DATA_DIR
            / "classification_train.csv"
        ),
        "classification_validation": (
            OUTPUT_DATA_DIR
            / "classification_validation.csv"
        ),
        "classification_test": (
            OUTPUT_DATA_DIR
            / "classification_test.csv"
        ),
        "regression_train": (
            OUTPUT_DATA_DIR
            / "regression_train.csv"
        ),
        "regression_validation": (
            OUTPUT_DATA_DIR
            / "regression_validation.csv"
        ),
        "regression_test": (
            OUTPUT_DATA_DIR
            / "regression_test.csv"
        ),
    }

    save_split(
        classification_train,
        paths["classification_train"],
    )

    save_split(
        classification_validation,
        paths["classification_validation"],
    )

    save_split(
        classification_test,
        paths["classification_test"],
    )

    save_split(
        regression_train,
        paths["regression_train"],
    )

    save_split(
        regression_validation,
        paths["regression_validation"],
    )

    save_split(
        regression_test,
        paths["regression_test"],
    )

    log_class_distribution(
        classification_train,
        "Classification train",
    )

    log_class_distribution(
        classification_validation,
        "Classification validation",
    )

    log_class_distribution(
        classification_test,
        "Classification test",
    )

    logger.info(
        "Model-development dataset shape: %s",
        modeling_data.shape,
    )

    return paths


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    generated_paths = create_data_splits()

    print("\n" + "=" * 60)
    print("DATA SPLITTING REPORT")
    print("=" * 60)

    for name, path in generated_paths.items():
        print(f"{name}:")
        print(f"  {path}")