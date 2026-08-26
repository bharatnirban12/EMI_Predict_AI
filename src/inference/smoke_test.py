"""
Inference smoke test.

Uses one existing test-dataset record to verify the complete
classification and regression prediction pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.inference.predictor import EMIPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification_test.csv"
)


def main() -> None:
    """Run a single-record inference smoke test."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print()
    print("=" * 70)
    print("EMI PREDICTOR INFERENCE SMOKE TEST")
    print("=" * 70)

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    test_data = pd.read_csv(TEST_DATA_PATH)

    print(f"Test dataset shape: {test_data.shape}")

    predictor = EMIPredictor()

    sample = test_data.iloc[[0]].copy()

    actual_classification = None
    actual_regression = None

    if "emi_eligibility" in sample.columns:
        actual_classification = sample[
            "emi_eligibility"
        ].iloc[0]

    if "max_monthly_emi" in sample.columns:
        actual_regression = sample[
            "max_monthly_emi"
        ].iloc[0]

    predictions = predictor.predict(sample)

    print()
    print("-" * 70)
    print("INFERENCE RESULT")
    print("-" * 70)

    print(
        "Predicted EMI eligibility:",
        predictions["emi_eligibility"].iloc[0],
    )

    print(
        "Predicted maximum monthly EMI:",
        predictions["max_monthly_emi"].iloc[0],
    )

    print()
    print("-" * 70)
    print("REFERENCE TARGETS FROM TEST RECORD")
    print("-" * 70)

    print(
        "Actual EMI eligibility:",
        actual_classification,
    )

    print(
        "Actual maximum monthly EMI:",
        actual_regression,
    )

    print()
    print("=" * 70)
    print("INFERENCE SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()