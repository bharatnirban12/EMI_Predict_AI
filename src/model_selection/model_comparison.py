"""
Model comparison and selection for the EMI prediction project.

This module compares the baseline classification and regression
models using the evaluation results established during model training.

No model retraining is performed by this module.

Outputs
-------
artifacts/model_selection/classification_model_comparison.csv
artifacts/model_selection/regression_model_comparison.csv
artifacts/model_selection/model_selection_summary.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_SELECTION_DIR = ARTIFACTS_DIR / "model_selection"


CLASSIFICATION_OUTPUT = (
    MODEL_SELECTION_DIR
    / "classification_model_comparison.csv"
)

REGRESSION_OUTPUT = (
    MODEL_SELECTION_DIR
    / "regression_model_comparison.csv"
)

SUMMARY_OUTPUT = (
    MODEL_SELECTION_DIR
    / "model_selection_summary.json"
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - %(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Baseline results
# ---------------------------------------------------------------------

CLASSIFICATION_RESULTS = [
    {
        "model": "Logistic Regression",
        "accuracy": 0.9480,
        "weighted_precision": 0.9382,
        "weighted_recall": 0.9480,
        "weighted_f1": 0.9402,
        "macro_f1": 0.74,
        "high_risk_recall": 0.24,
    },
    {
        "model": "Random Forest",
        "accuracy": 0.9485,
        "weighted_precision": 0.9360,
        "weighted_recall": 0.9485,
        "weighted_f1": 0.9320,
        "macro_f1": 0.6764,
        "high_risk_recall": 0.07,
    },
    {
        "model": "XGBoost",
        "accuracy": 0.9713,
        "weighted_precision": 0.9687,
        "weighted_recall": 0.9713,
        "weighted_f1": 0.9682,
        "macro_f1": 0.8567,
        "high_risk_recall": 0.50,
    },
    {
        "model": "LightGBM",
        "accuracy": 0.9737,
        "weighted_precision": 0.9716,
        "weighted_recall": 0.9737,
        "weighted_f1": 0.9719,
        "macro_f1": 0.8756,
        "high_risk_recall": 0.5793401413982718,
    },
    {
        "model": "CatBoost",
        "accuracy": 0.9634,
        "weighted_precision": 0.9599,
        "weighted_recall": 0.9634,
        "weighted_f1": 0.9549,
        "macro_f1": 0.7817,
        "high_risk_recall": 0.27,
    },
]


REGRESSION_RESULTS = [
    {
        "model": "Linear Regression",
        "mae": 1856.0064,
        "rmse": 2583.0998,
        "r2": 0.8831,
    },
    {
        "model": "Random Forest",
        "mae": 263.2739,
        "rmse": 838.5793,
        "r2": 0.9877,
    },
    {
        "model": "XGBoost",
        "mae": 246.006840633374,
        "rmse": 534.1256429417401,
        "r2": 0.9950003920814388,
    },
    {
        "model": "LightGBM",
        "mae": 264.9135,
        "rmse": 538.1468,
        "r2": 0.9949,
    },
    {
        "model": "CatBoost",
        "mae": 323.8480,
        "rmse": 648.0812,
        "r2": 0.9926,
    },
]


# ---------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------

def select_classification_model(
    dataframe: pd.DataFrame,
) -> tuple[str, str]:
    """
    Select the classification candidate.

    Selection is based on the strongest recorded overall metrics,
    with macro F1 and High_Risk recall considered alongside accuracy
    and weighted F1.

    Parameters
    ----------
    dataframe:
        Classification comparison dataframe.

    Returns
    -------
    tuple[str, str]
        Selected model and selection rationale.
    """
    selected_model = "LightGBM"

    rationale = (
        "LightGBM has the strongest recorded test accuracy, "
        "weighted F1, macro F1, and High_Risk recall among "
        "the evaluated classification baselines."
    )

    if selected_model not in set(
        dataframe["model"]
    ):
        raise ValueError(
            "Selected classification model is "
            "missing from comparison results."
        )

    return selected_model, rationale


def select_regression_model(
    dataframe: pd.DataFrame,
) -> tuple[str, str]:
    """
    Select the regression candidate.

    Selection prioritizes MAE and RMSE while also considering R².

    Parameters
    ----------
    dataframe:
        Regression comparison dataframe.

    Returns
    -------
    tuple[str, str]
        Selected model and selection rationale.
    """
    selected_model = "XGBoost"

    rationale = (
        "XGBoost has the lowest recorded test MAE and RMSE "
        "and the highest recorded R² among the evaluated "
        "regression baselines."
    )

    if selected_model not in set(
        dataframe["model"]
    ):
        raise ValueError(
            "Selected regression model is "
            "missing from comparison results."
        )

    return selected_model, rationale


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_regression_requirements(
    dataframe: pd.DataFrame,
    selected_model: str,
) -> dict[str, bool]:
    """
    Validate the established regression requirements.

    Requirements
    ------------
    MAE < 500
    R² > 0.85
    """
    selected_row = dataframe.loc[
        dataframe["model"] == selected_model
    ]

    if selected_row.empty:
        raise ValueError(
            "Selected regression model was not found."
        )

    mae = float(
        selected_row.iloc[0]["mae"]
    )

    r2 = float(
        selected_row.iloc[0]["r2"]
    )

    return {
        "mae_less_than_500": mae < 500,
        "r2_greater_than_0_85": r2 > 0.85,
    }


# ---------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------

def run_model_comparison() -> None:
    """
    Run model comparison and generate selection artifacts.
    """
    logger.info(
        "Starting model comparison and selection."
    )

    MODEL_SELECTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    classification_df = pd.DataFrame(
        CLASSIFICATION_RESULTS
    )

    regression_df = pd.DataFrame(
        REGRESSION_RESULTS
    )

    classification_df[
        "selected"
    ] = classification_df["model"].eq(
        "LightGBM"
    )

    regression_df[
        "selected"
    ] = regression_df["model"].eq(
        "XGBoost"
    )

    classification_model, classification_reason = (
        select_classification_model(
            classification_df
        )
    )

    regression_model, regression_reason = (
        select_regression_model(
            regression_df
        )
    )

    regression_requirements = (
        validate_regression_requirements(
            regression_df,
            regression_model,
        )
    )

    classification_df.to_csv(
        CLASSIFICATION_OUTPUT,
        index=False,
    )

    regression_df.to_csv(
        REGRESSION_OUTPUT,
        index=False,
    )

    summary = {
        "classification": {
            "selected_model": classification_model,
            "selection_basis": (
                "Overall test performance with "
                "particular consideration of macro F1 "
                "and High_Risk recall."
            ),
            "rationale": classification_reason,
            "test_metrics": (
                classification_df.loc[
                    classification_df["model"]
                    == classification_model
                ]
                .iloc[0]
                .to_dict()
            ),
            "known_limitation": (
                "High_Risk recall is 0.5793, meaning "
                "1,071 of 2,546 actual High_Risk test "
                "records were missed."
            ),
        },
        "regression": {
            "selected_model": regression_model,
            "selection_basis": (
                "Lowest test MAE and RMSE with "
                "highest recorded R2."
            ),
            "rationale": regression_reason,
            "test_metrics": (
                regression_df.loc[
                    regression_df["model"]
                    == regression_model
                ]
                .iloc[0]
                .to_dict()
            ),
            "requirements": regression_requirements,
        },
        "artifacts": {
            "classification_comparison": str(
                CLASSIFICATION_OUTPUT
            ),
            "regression_comparison": str(
                REGRESSION_OUTPUT
            ),
        },
    }

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    logger.info(
        "Classification selected model: %s",
        classification_model,
    )

    logger.info(
        "Regression selected model: %s",
        regression_model,
    )

    logger.info(
        "Classification comparison saved: %s",
        CLASSIFICATION_OUTPUT,
    )

    logger.info(
        "Regression comparison saved: %s",
        REGRESSION_OUTPUT,
    )

    logger.info(
        "Model selection summary saved: %s",
        SUMMARY_OUTPUT,
    )

    print()
    print("=" * 70)
    print("MODEL SELECTION SUMMARY")
    print("=" * 70)

    print()
    print("Classification")
    print("-" * 70)
    print(
        f"Selected model: {classification_model}"
    )
    selected_classification = classification_df.loc[
        classification_df["model"]
        == classification_model
    ].iloc[0]

    print(
        f"Accuracy:       "
        f"{selected_classification['accuracy']:.4f}"
    )
    print(
        f"Weighted F1:    "
        f"{selected_classification['weighted_f1']:.4f}"
    )
    print(
        f"Macro F1:       "
        f"{selected_classification['macro_f1']:.4f}"
    )
    print(
        f"High_Risk Recall: "
        f"{selected_classification['high_risk_recall']:.4f}"
    )

    print()
    print("Regression")
    print("-" * 70)
    print(
        f"Selected model: {regression_model}"
    )

    selected_regression = regression_df.loc[
        regression_df["model"]
        == regression_model
    ].iloc[0]

    print(
        f"MAE:            "
        f"{selected_regression['mae']:.4f}"
    )
    print(
        f"RMSE:           "
        f"{selected_regression['rmse']:.4f}"
    )
    print(
        f"R2:             "
        f"{selected_regression['r2']:.4f}"
    )

    print()
    print("Regression Requirements")
    print("-" * 70)
    print(
        "MAE < 500:       "
        f"{'PASS' if regression_requirements['mae_less_than_500'] else 'FAIL'}"
    )
    print(
        "R2 > 0.85:       "
        f"{'PASS' if regression_requirements['r2_greater_than_0_85'] else 'FAIL'}"
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_model_comparison()