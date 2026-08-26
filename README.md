# EMIPredict AI

EMIPredict AI is an end-to-end machine learning system that predicts EMI eligibility and maximum affordable monthly EMI using tuned LightGBM and XGBoost models.

## Project Overview

The system provides two complementary predictions for an EMI/loan application:

1. **EMI Eligibility Classification**
   - Model: Tuned LightGBM
   - Classes:
     - `Eligible`
     - `High_Risk`
     - `Not_Eligible`

2. **Maximum Monthly EMI Regression**
   - Model: Tuned XGBoost
   - Target: `max_monthly_emi`

The application is designed with a Streamlit frontend and FastAPI backend.

## Model Performance

### Classification — Tuned LightGBM

Test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 0.9787 |
| Weighted Precision | 0.9778 |
| Weighted Recall | 0.9787 |
| Weighted F1 | 0.9781 |
| Macro F1 | 0.9055 |
| High_Risk Recall | 0.7090 |

Final configuration:

```text
n_estimators = 400
learning_rate = 0.08
num_leaves = 63
max_depth = 10
min_child_samples = 50
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

### Regression — Tuned XGBoost

Test-set results:

| Metric | Score |
|---|---:|
| MAE | 185.7806 |
| RMSE | 454.5660 |
| R² | 0.9964 |

Final configuration:

```text
n_estimators = 400
max_depth = 7
learning_rate = 0.08
min_child_weight = 3
subsample = 0.8
colsample_bytree = 1.0
objective = reg:squarederror
random_state = 42
n_jobs = -1
```

Both models passed the project's stated validation requirements:

```text
Classification: Accuracy = 0.9787
Regression:      MAE < 500
Regression:      R² > 0.85
```

## Dataset

The project uses separate training, validation, and test datasets.

The recorded split sizes used during final modeling were:

```text
Training:   272,769 records
Validation:  58,451 records
Test:        58,451 records
```

The repository intentionally does not include the datasets. Dataset directories are excluded through `.gitignore`.

## Feature Engineering

The project applies reusable feature engineering before model inference.

Generated features include:

```text
total_education_expenses
total_monthly_living_expenses
total_monthly_expenses
disposable_income
expense_to_income_ratio
emi_to_income_ratio
requested_amount_to_income_ratio
requested_amount_per_month
```

The final preprocessing pipeline produces 56 transformed features for both production models.

## Explainability

Model explainability was performed using SHAP.

Important classification features identified during explainability include:

```text
requested_amount_per_month
disposable_income
requested_amount_to_income_ratio
house_type_Rented
expense_to_income_ratio
credit_score
current_emi_amount
bank_balance
monthly_rent
years_of_employment
```

High_Risk error analysis was also performed to investigate classification errors for the minority `High_Risk` class.

## Project Structure

```text
EMIPredict_AI/
│
├── artifacts/
│   ├── classification/
│   │   ├── lightgbm_tuned_model.pkl
│   │   ├── lightgbm_tuned_preprocessor.pkl
│   │   └── lightgbm_tuned_label_mapping.pkl
│   │
│   └── regression/
│       ├── xgboost_regressor_model.pkl
│       └── xgboost_regressor_preprocessor.pkl
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── analysis/
│   ├── evaluation/
│   ├── experiments/
│   ├── explainability/
│   ├── inference/
│   └── models/
│       ├── classification/
│       └── regression/
│
├── streamlit_app/
│   └── app.py
│
├── tests/
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Local Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd EMIPredict_AI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv myenv
myenv\Scripts\Activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Running the FastAPI Backend

Start the API using the project's FastAPI application module.

The production API exposes:

```text
GET  /health
POST /predict
GET  /docs
```

After starting the API, verify:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

Expected health response:

```json
{
  "status": "ok"
}
```

## Running Streamlit

The Streamlit application is located at:

```text
streamlit_app/app.py
```

Run:

```powershell
streamlit run streamlit_app/app.py
```

When using the FastAPI backend architecture, configure the Streamlit application with the deployed FastAPI URL through the `EMI_API_URL` environment variable.

Example:

```text
EMI_API_URL=https://YOUR-RENDER-URL.onrender.com
```

Do not commit `.env` files or API credentials.

## Example Prediction

Example request payload:

```json
{
  "age": 48,
  "gender": "Female",
  "marital_status": "Married",
  "education": "Graduate",
  "monthly_salary": 37800,
  "employment_type": "Private",
  "years_of_employment": 2.2,
  "company_type": "Startup",
  "house_type": "Family",
  "monthly_rent": 0,
  "family_size": 4,
  "dependents": 3,
  "school_fees": 14100,
  "college_fees": 0,
  "travel_expenses": 2800,
  "groceries_utilities": 6900,
  "other_monthly_expenses": 2800,
  "existing_loans": "No",
  "current_emi_amount": 0,
  "credit_score": 702,
  "bank_balance": 275300,
  "emergency_fund": 128300,
  "emi_scenario": "Education EMI",
  "requested_amount": 134000,
  "requested_tenure": 33
}
```

Example response format:

```json
{
  "emi_eligibility": "Eligible",
  "max_monthly_emi": 23256.271484375
}
```

The example response above is an observed test response from the project workflow; it is not presented as a guaranteed output for all inputs.

## Deployment

The intended deployment architecture is:

```text
User
  |
  v
Streamlit Cloud
  |
  | HTTPS
  v
Render
  |
  v
FastAPI
  |
  v
EMIPredictor
  |
  +-------------------+
  |                   |
  v                   v
LightGBM           XGBoost
Classifier         Regressor
  |                   |
  v                   v
EMI Eligibility   Maximum Monthly EMI
```

### Streamlit Cloud

Application entrypoint:

```text
streamlit_app/app.py
```

Configure the FastAPI backend URL using:

```text
EMI_API_URL
```

### Render

Deploy the FastAPI application as a web service.

The service must listen on:

```text
0.0.0.0
```

and use the platform-provided `PORT` environment variable.

Example command pattern:

```bash
uvicorn <FASTAPI_MODULE>:app --host 0.0.0.0 --port $PORT
```

Replace `<FASTAPI_MODULE>` with the actual module containing the FastAPI `app` object.

## Production Artifacts

The production inference artifacts are tracked in Git because the deployed FastAPI service requires them:

```text
artifacts/classification/
artifacts/regression/
```

Large datasets, local environments, caches, logs, and temporary files are excluded through `.gitignore`.

## Validation and Evaluation

The project includes dedicated evaluation outputs for:

- Classification metrics
- Regression metrics
- Confusion matrix
- Classification error analysis
- Regression error analysis
- Feature importance
- SHAP explainability
- High_Risk classification error analysis

## Development Notes

The project follows a modular structure separating:

- Data processing
- Feature engineering
- Model training
- Hyperparameter tuning
- Evaluation
- Explainability
- Error analysis
- Inference
- Application deployment

Production model artifacts should be regenerated only when the corresponding training pipeline is intentionally rerun.

