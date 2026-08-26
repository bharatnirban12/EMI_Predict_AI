"""
Module: load_data.py

Description:
    Utility functions for loading the EMI Prediction dataset.

Author:
    EMIPredict AI Project

"""

from pathlib import Path
import logging

import pandas as pd


# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.

    ValueError
        If the CSV file is empty.

    Exception
        For any unexpected loading errors.
    """

    path = Path(file_path)

    if not path.exists():
        logger.error("Dataset not found: %s", file_path)
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'."
        )

    try:
        df = pd.read_csv(path)

        if df.empty:
            logger.error("Dataset is empty.")
            raise ValueError("Dataset is empty.")

        logger.info("Dataset loaded successfully.")
        logger.info("Rows: %d", df.shape[0])
        logger.info("Columns: %d", df.shape[1])

        return df

    except pd.errors.EmptyDataError as exc:
        logger.exception("CSV file is empty.")
        raise ValueError("CSV file is empty.") from exc

    except Exception as exc:
        logger.exception("Unexpected error while loading dataset.")
        raise exc


if __name__ == "__main__":
    DATA_PATH = "data/raw/emi_prediction_dataset.csv"

    dataframe = load_dataset(DATA_PATH)

    print("\nDataset Shape:")
    print(dataframe.shape)

    print("\nFirst Five Rows:")
    print(dataframe.head())