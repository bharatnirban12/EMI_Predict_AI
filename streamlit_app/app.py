"""
Streamlit frontend for the EMI Predictor API.

The application collects the 25 raw model inputs and sends them
to the FastAPI /predict endpoint.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = "https://emi-predict-ai-27dj.onrender.com"

API_URL = os.getenv(
    "EMI_API_URL",
    DEFAULT_API_URL,
).rstrip("/")


st.set_page_config(
    page_title="EMI Predictor",
    page_icon="💰",
    layout="wide",
)


st.title("EMI Predictor")
st.markdown(
    "Enter applicant and loan information to predict "
    "EMI eligibility and maximum monthly EMI."
)


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Send prediction request to the FastAPI service.

    Parameters
    ----------
    payload:
        Dictionary containing the 25 required raw input fields.

    Returns
    -------
    dict
        Prediction response from the API.

    Raises
    ------
    requests.RequestException
        If the API cannot be reached.
    ValueError
        If the API returns an unsuccessful response.
    """

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=30,
    )

    if not response.ok:
        raise ValueError(
            f"API request failed with status "
            f"{response.status_code}: {response.text}"
        )

    return response.json()


with st.sidebar:
    st.header("API Configuration")

    st.code(
        API_URL,
        language="text",
    )

    st.caption(
        "The Streamlit application sends predictions "
        "to the FastAPI service."
    )


st.subheader("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18.0,
        max_value=100.0,
        value=32.0,
        step=1.0,
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"],
    )

    education = st.selectbox(
        "Education",
        [
            "Graduate",
            "High School",
            "Post Graduate",
            "Professional",
        ],
    )

with col2:
    employment_type = st.selectbox(
        "Employment Type",
        [
            "Government",
            "Private",
            "Self-employed",
        ],
    )

    years_of_employment = st.number_input(
        "Years of Employment",
        min_value=0.0,
        max_value=60.0,
        value=0.6,
        step=0.1,
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "Large Indian",
            "MNC",
            "Mid-size",
            "Small",
            "Startup",
        ],
    )

    house_type = st.selectbox(
        "House Type",
        [
            "Family",
            "Own",
            "Rented",
        ],
    )

with col3:
    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=20,
        value=4,
        step=1,
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
    )

st.subheader("Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    monthly_salary = st.number_input(
        "Monthly Salary",
        min_value=0.0,
        value=45500.0,
        step=500.0,
    )

    monthly_rent = st.number_input(
        "Monthly Rent",
        min_value=0.0,
        value=0.0,
        step=500.0,
    )

    school_fees = st.number_input(
        "School Fees",
        min_value=0.0,
        value=0.0,
        step=500.0,
    )

    college_fees = st.number_input(
        "College Fees",
        min_value=0.0,
        value=0.0,
        step=500.0,
    )

with col2:
    travel_expenses = st.number_input(
        "Travel Expenses",
        min_value=0.0,
        value=2600.0,
        step=100.0,
    )

    groceries_utilities = st.number_input(
        "Groceries & Utilities",
        min_value=0.0,
        value=8400.0,
        step=100.0,
    )

    other_monthly_expenses = st.number_input(
        "Other Monthly Expenses",
        min_value=0.0,
        value=3600.0,
        step=100.0,
    )

    existing_loans = st.selectbox(
        "Existing Loans",
        ["No", "Yes"],
    )

with col3:
    current_emi_amount = st.number_input(
        "Current EMI Amount",
        min_value=0.0,
        value=0.0,
        step=500.0,
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0.0,
        max_value=1000.0,
        value=637.0,
        step=1.0,
    )

    bank_balance = st.number_input(
        "Bank Balance",
        min_value=0.0,
        value=300100.0,
        step=1000.0,
    )

    emergency_fund = st.number_input(
        "Emergency Fund",
        min_value=0.0,
        value=78900.0,
        step=1000.0,
    )


st.subheader("EMI Request")

col1, col2, col3 = st.columns(3)

with col1:
    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "E-commerce Shopping EMI",
            "Education EMI",
            "Home Appliances EMI",
            "Personal Loan EMI",
            "Vehicle EMI",
        ],
    )

with col2:
    requested_amount = st.number_input(
        "Requested Amount",
        min_value=0.0,
        value=435000.0,
        step=1000.0,
    )

with col3:
    requested_tenure = st.number_input(
        "Requested Tenure (In months)",
        min_value=1,
        max_value=120,
        value=21,
        step=1,
    )


payload: dict[str, Any] = {
    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "education": education,
    "monthly_salary": monthly_salary,
    "employment_type": employment_type,
    "years_of_employment": years_of_employment,
    "company_type": company_type,
    "house_type": house_type,
    "monthly_rent": monthly_rent,
    "family_size": family_size,
    "dependents": dependents,
    "school_fees": school_fees,
    "college_fees": college_fees,
    "travel_expenses": travel_expenses,
    "groceries_utilities": groceries_utilities,
    "other_monthly_expenses": other_monthly_expenses,
    "existing_loans": existing_loans,
    "current_emi_amount": current_emi_amount,
    "credit_score": credit_score,
    "bank_balance": bank_balance,
    "emergency_fund": emergency_fund,
    "emi_scenario": emi_scenario,
    "requested_amount": requested_amount,
    "requested_tenure": requested_tenure,
}


st.divider()

if st.button(
    "Predict EMI Eligibility",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Calling EMI prediction API..."):
        try:
            result = predict(payload)

        except requests.RequestException as exc:
            st.error(
                "Could not connect to the FastAPI service."
            )
            st.exception(exc)

        except ValueError as exc:
            st.error("Prediction API returned an error.")
            st.exception(exc)

        else:
            eligibility = result["emi_eligibility"]
            maximum_emi = float(
                result["max_monthly_emi"]
            )

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "EMI Eligibility",
                    eligibility,
                )

            with result_col2:
                st.metric(
                    "Maximum Monthly EMI",
                    f"₹{maximum_emi:,.2f}",
                )

            st.success(
                "Prediction completed successfully."
            )
