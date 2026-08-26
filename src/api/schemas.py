"""
Pydantic schemas for the EMI Predictor API.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EMIRequest(BaseModel):
    """
    Request payload containing all raw features required for prediction.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    age: int
    gender: str
    marital_status: str
    education: str
    monthly_salary: float
    employment_type: str
    years_of_employment: float
    company_type: str
    house_type: str
    monthly_rent: float
    family_size: int
    dependents: int
    school_fees: float
    college_fees: float
    travel_expenses: float
    groceries_utilities: float
    other_monthly_expenses: float
    existing_loans: str
    current_emi_amount: float
    credit_score: int
    bank_balance: float
    emergency_fund: float
    emi_scenario: str
    requested_amount: float
    requested_tenure: int


class EMIResponse(BaseModel):
    """
    Prediction response containing both classification
    and regression predictions.
    """

    emi_eligibility: str
    max_monthly_emi: float