"""
Module: save_cleaned_data.py

Description:
    Execute the data-cleaning pipeline and persist the resulting
    cleaned dataset and domain-violation report.

Author:
    EMIPredict AI Project
"""

from __future__ import annotations

import logging
from pathlib import Path

from clean_data import clean_dataset
from load_data import load_dataset


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "emi_prediction_dataset.csv"
)

INTERIM_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

CLEANED_DATA_PATH = (
    INTERIM_DATA_DIR
    / "cleaned_data.csv"
)

DOMAIN_VIOLATIONS_PATH = (
    INTERIM_DATA_DIR
    / "domain_violations.csv"
)


def save_cleaned_data(
    cleaned_data_path: Path = CLEANED_DATA_PATH,
    violations_path: Path = DOMAIN_VIOLATIONS_PATH,
) -> tuple[Path, Path]:
    """
    Execute the cleaning pipeline and save its outputs.

    Parameters
    ----------
    cleaned_data_path : Path
        Destination for the cleaned dataset.

    violations_path : Path
        Destination for the domain-violation report.

    Returns
    -------
    tuple[Path, Path]
        Paths to the saved cleaned dataset and violation report.

    Raises
    ------
    FileNotFoundError
        If the raw dataset does not exist.

    RuntimeError
        If either output file cannot be verified after saving.
    """
    logger.info("Starting cleaned-data generation.")

    cleaned_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    violations_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Loading raw dataset from: %s",
        RAW_DATA_PATH,
    )

    dataframe = load_dataset(
        str(RAW_DATA_PATH)
    )

    cleaned_data, domain_violations = clean_dataset(
        dataframe
    )

    logger.info(
        "Saving cleaned dataset to: %s",
        cleaned_data_path,
    )

    cleaned_data.to_csv(
        cleaned_data_path,
        index=False,
    )

    logger.info(
        "Saving domain-violation report to: %s",
        violations_path,
    )

    domain_violations.to_csv(
        violations_path,
        index=True,
        index_label="original_row_index",
    )

    if not cleaned_data_path.exists():
        raise RuntimeError(
            "Cleaned dataset was not created successfully."
        )

    if not violations_path.exists():
        raise RuntimeError(
            "Domain-violation report was not created successfully."
        )

    logger.info(
        "Cleaned dataset saved successfully."
    )

    logger.info(
        "Domain-violation report saved successfully."
    )

    return cleaned_data_path, violations_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
    )

    cleaned_path, violations_path = save_cleaned_data()

    print("\n" + "=" * 60)
    print("DATA SAVING REPORT")
    print("=" * 60)

    print("\nCleaned dataset:")
    print(cleaned_path)

    print("\nDomain-violation report:")
    print(violations_path)

    print("\nFiles created successfully:")
    print(f"Cleaned dataset exists: {cleaned_path.exists()}")
    print(
        "Violation report exists: "
        f"{violations_path.exists()}"
    )