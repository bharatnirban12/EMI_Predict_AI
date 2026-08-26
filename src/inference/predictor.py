"""
Production inference layer for the EMI prediction project.

This module loads the selected classification and regression models
and their fitted preprocessing artifacts.

Selected models
---------------
Classification:
    LightGBM

Regression:
    XGBoost

The module does not retrain models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn.compose._column_transformer

if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CLASSIFICATION_MODEL_PATH = (
    ARTIFACTS_DIR
    / "classification"
    / "lightgbm_tuned_model.pkl"
)

CLASSIFICATION_PREPROCESSOR_PATH = (
    ARTIFACTS_DIR
    / "classification"
    / "lightgbm_tuned_preprocessor.pkl"
)

CLASSIFICATION_LABEL_MAPPING_PATH = (
    ARTIFACTS_DIR
    / "classification"
    / "lightgbm_tuned_label_mapping.pkl"
)

REGRESSION_MODEL_PATH = (
    ARTIFACTS_DIR
    / "regression"
    / "xgboost_regressor_model.pkl"
)

REGRESSION_PREPROCESSOR_PATH = (
    ARTIFACTS_DIR
    / "regression"
    / "xgboost_regressor_preprocessor.pkl"
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------

RAW_FEATURE_COLUMNS = [
    "age",
    "gender",
    "marital_status",
    "education",
    "monthly_salary",
    "employment_type",
    "years_of_employment",
    "company_type",
    "house_type",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "existing_loans",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "emi_scenario",
    "requested_amount",
    "requested_tenure",
]


CLASSIFICATION_TARGET = "emi_eligibility"

REGRESSION_TARGET = "max_monthly_emi"


ENGINEERED_FEATURE_COLUMNS = [
    "total_education_expenses",
    "total_monthly_living_expenses",
    "total_monthly_expenses",
    "disposable_income",
    "expense_to_income_ratio",
    "emi_to_income_ratio",
    "requested_amount_to_income_ratio",
    "requested_amount_per_month",
]


EXPECTED_TRANSFORMED_FEATURE_COUNT = 56

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def add_engineered_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the engineered features used by model development.

    Parameters
    ----------
    dataframe:
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe containing engineered features.
    """
    data = dataframe.copy()

    required_columns = [
        "monthly_salary",
        "school_fees",
        "college_fees",
        "monthly_rent",
        "groceries_utilities",
        "travel_expenses",
        "other_monthly_expenses",
        "current_emi_amount",
        "requested_amount",
        "requested_tenure",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns required for feature engineering: "
            f"{missing_columns}"
        )

    data["total_education_expenses"] = (
        data["school_fees"]
        + data["college_fees"]
    )

    data["total_monthly_living_expenses"] = (
        data["monthly_rent"]
        + data["groceries_utilities"]
        + data["travel_expenses"]
        + data["other_monthly_expenses"]
    )

    data["total_monthly_expenses"] = (
        data["total_education_expenses"]
        + data["total_monthly_living_expenses"]
    )

    data["disposable_income"] = (
        data["monthly_salary"]
        - data["total_monthly_expenses"]
    )

    data["expense_to_income_ratio"] = (
        data["total_monthly_expenses"]
        / data["monthly_salary"].replace(0, pd.NA)
    )

    data["emi_to_income_ratio"] = (
        data["current_emi_amount"]
        / data["monthly_salary"].replace(0, pd.NA)
    )

    data["requested_amount_to_income_ratio"] = (
        data["requested_amount"]
        / data["monthly_salary"].replace(0, pd.NA)
    )

    data["requested_amount_per_month"] = (
        data["requested_amount"]
        / data["requested_tenure"].replace(0, pd.NA)
    )

    data[ENGINEERED_FEATURE_COLUMNS] = (
        data[ENGINEERED_FEATURE_COLUMNS]
        .astype(float)
    )

    return data


# ---------------------------------------------------------------------
# Predictor preparation
# ---------------------------------------------------------------------

def prepare_predictors(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform feature engineering and remove target columns.

    Parameters
    ----------
    dataframe:
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Predictor dataframe.
    """
    data = add_engineered_features(dataframe)

    excluded_targets = {
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET,
    }

    predictor_columns = [
        column
        for column in data.columns
        if column not in excluded_targets
    ]

    predictors = data[predictor_columns].copy()

    return predictors


# ---------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------

def validate_artifacts(
    classification_model: Any,
    classification_preprocessor: Any,
    regression_model: Any,
    regression_preprocessor: Any,
) -> None:
    """
    Validate the expected serialized artifact structure.

    Raises
    ------
    TypeError
        If an artifact has an unexpected structure.
    ValueError
        If required preprocessor components are missing.
    """
    classification_transformer = (
        classification_preprocessor
    )

    if not hasattr(
        classification_transformer,
        "transform",
    ):
        raise TypeError(
            "Classification 'preprocessor' does not "
            "provide transform()."
        )

    if not hasattr(
        regression_preprocessor,
        "transform",
    ):
        raise TypeError(
            "Regression preprocessor does not "
            "provide transform()."
        )

    if not hasattr(
        classification_model,
        "predict",
    ):
        raise TypeError(
            "Classification model does not provide predict()."
        )

    if not hasattr(
        regression_model,
        "predict",
    ):
        raise TypeError(
            "Regression model does not provide predict()."
        )

    logger.info(
        "Model and preprocessing artifacts validated."
    )


# ---------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------

class EMIPredictor:
    """
    Reusable inference engine for EMI predictions.
    """

    def __init__(
        self,
        classification_model_path: Path = (
            CLASSIFICATION_MODEL_PATH
        ),
        classification_preprocessor_path: Path = (
            CLASSIFICATION_PREPROCESSOR_PATH
        ),
        
        classification_label_mapping_path: Path = (
            CLASSIFICATION_LABEL_MAPPING_PATH
        ),
        
        regression_model_path: Path = (
            REGRESSION_MODEL_PATH
        ),
        regression_preprocessor_path: Path = (
            REGRESSION_PREPROCESSOR_PATH
        ),
    ) -> None:
        """
        Load production model artifacts.

        Parameters
        ----------
        classification_model_path:
            Path to the LightGBM classifier.

        classification_preprocessor_path:
            Path to the LightGBM preprocessing artifact.

        regression_model_path:
            Path to the XGBoost regressor.

        regression_preprocessor_path:
            Path to the XGBoost preprocessing artifact.
        """
        self.classification_model = joblib.load(
            classification_model_path
        )

        self.classification_artifacts = joblib.load(
            classification_preprocessor_path
        )

        self.label_mapping = joblib.load(
            classification_label_mapping_path
        )

        self.regression_model = joblib.load(
            regression_model_path
        )

        self.regression_preprocessor = joblib.load(
            regression_preprocessor_path
        )

        validate_artifacts(
            classification_model=self.classification_model,
            classification_preprocessor=(
                self.classification_artifacts
            ),
            regression_model=self.regression_model,
            regression_preprocessor=(
                self.regression_preprocessor
            ),
        )

        self.classification_preprocessor = (
            self.classification_artifacts
        )

        self.reverse_label_mapping = {
            value: key
            for key, value in self.label_mapping.items()
        }

        self._validate_feature_contracts()

        logger.info(
            "EMIPredictor initialized successfully."
        )

    def _validate_feature_contracts(self) -> None:
        """
        Validate transformed feature counts.
        """
        classification_features = (
            self.classification_preprocessor
            .get_feature_names_out()
        )

        regression_features = (
            self.regression_preprocessor
            .get_feature_names_out()
        )

        if len(classification_features) != (
            EXPECTED_TRANSFORMED_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected classification transformed "
                f"feature count: {len(classification_features)}"
            )

        if len(regression_features) != (
            EXPECTED_TRANSFORMED_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected regression transformed "
                f"feature count: {len(regression_features)}"
            )

        if len(classification_features) != len(
            regression_features
        ):
            raise ValueError(
                "Classification and regression transformed "
                "feature counts do not match."
            )

        logger.info(
            "Feature contract validation passed: %d features.",
            EXPECTED_TRANSFORMED_FEATURE_COUNT,
        )

    def _validate_input_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate required raw input columns.
        """
        missing_columns = [
            column
            for column in RAW_FEATURE_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required input columns: "
                f"{missing_columns}"
            )

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate classification and regression predictions.

        Parameters
        ----------
        dataframe:
            Dataframe containing the required raw input columns.

        Returns
        -------
        pd.DataFrame
            Prediction results.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Input dataframe must not be empty."
            )

        self._validate_input_columns(dataframe)

        predictors = prepare_predictors(
            dataframe
        )

        logger.info(
            "Prepared predictors: %s",
            predictors.shape,
        )

        classification_transformed = (
            self.classification_preprocessor.transform(
                predictors
            )
        )

        regression_transformed = (
            self.regression_preprocessor.transform(
                predictors
            )
        )

        if classification_transformed.shape[1] != (
            EXPECTED_TRANSFORMED_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected classification transformed "
                f"shape: {classification_transformed.shape}"
            )

        if regression_transformed.shape[1] != (
            EXPECTED_TRANSFORMED_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected regression transformed "
                f"shape: {regression_transformed.shape}"
            )

        encoded_predictions = (
            self.classification_model.predict(
                classification_transformed
            )
        )

        encoded_predictions = [
            int(value)
            for value in encoded_predictions
        ]

        classification_predictions = [
            self.reverse_label_mapping[value]
            for value in encoded_predictions
        ]

        regression_predictions = (
            self.regression_model.predict(
                regression_transformed
            )
        )

        results = pd.DataFrame(
            {
                "emi_eligibility": (
                    classification_predictions
                ),
                "max_monthly_emi": (
                    regression_predictions
                ),
            },
            index=dataframe.index,
        )

        results["max_monthly_emi"] = (
            results["max_monthly_emi"]
            .astype(float)
        )

        logger.info(
            "Generated predictions for %d records.",
            len(results),
        )

        return results


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------

def main() -> None:
    """
    Validate artifact loading and inference initialization.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - %(message)s"
        ),
    )

    predictor = EMIPredictor()

    print()
    print("=" * 70)
    print("EMI PREDICTOR INITIALIZATION")
    print("=" * 70)
    print(
        "Classification model:",
        type(predictor.classification_model),
    )
    print(
        "Regression model:",
        type(predictor.regression_model),
    )
    print(
        "Classification transformed features:",
        len(
            predictor.classification_preprocessor
            .get_feature_names_out()
        ),
    )
    print(
        "Regression transformed features:",
        len(
            predictor.regression_preprocessor
            .get_feature_names_out()
        ),
    )
    print(
        "Label mapping:",
        predictor.label_mapping,
    )
    print("=" * 70)


if __name__ == "__main__":
    main()