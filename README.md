# Fraud Detection API — Bank Account Application Classification

## Overview
In today’s world, fraud in the financial sector remains one of the most pressing issues.  
Fraudulent activities cause significant damage to both businesses and individuals, making timely detection and prevention highly important in practice.

This project focuses on **classifying bank account applications** to detect potentially fraudulent ones using **machine learning models** trained on the [Bank Account Fraud Dataset Suite (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data).  
The dataset contains realistic and meaningful (non-encoded) features, enabling both predictive modeling and detailed exploratory analysis.

---

## Project Structure
```
├── api
│   ├── app
│   │   ├── database
│   │   │   ├── db.py
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   └── requests.py
│   │   ├── estimator
│   │   │   └── model.pkl         # Trained model
│   │   ├── config
│   │   │   └── conf.py           # Getting keys from .env
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── application.py            # FastAPI app entry point
│   ├── .env.example              # example .env file
│   ├── __init__.py
│   └── logger_config.py
├── data
│   ├── Bank Account Fraud Dataset
│   │   └── account_fraud.csv     # Original dataset
│   └── EDA
│       └── Fraud_EDA.ipynb       # Exploratory Data Analysis
├── Dockerfile
├── poetry.lock
├── pyproject.toml
├── README.md
└── training
    ├── dataleakage.ipynb          # Experiment with data leakage (not for production)
    ├── DataTransformer.py         # Data preprocessing classes
    ├── __init__.py
    ├── optuna
    │   └── optuna_study.db
    └── training.ipynb             # Model training & tuning
```

---

## EDA & Feature Engineering Summary
- **Train/test split**: Months 6–7 → test (80/20 split)
- **Target imbalance**: Fraud cases ≈ 1% of data
- **Removed constant/irrelevant features**:
  - `device_fraud_count` (handled outside the model — if >0, classify as fraud)
  - `zip_count_4w`, `source`, `bank_months_count`, `minutes_since_request`
- **Feature transformations**:
  - `proposed_credit_limit` converted to categorical (3 bins)
  - Rare categories merged into `Other`
- **Feature selection**: Correlation analysis, mutual information, feature importance

---

## Model Training Summary
- **Main metric**: Recall@5% FPR  
  (Maximize fraud detection while keeping false positives ≤ 5%)
- **Approach**:
  - Train baseline model
  - Train advanced models (RandomForest, XGBoost)
  - Time Series cross-validation by `Month`
  - Select 2 best models → perform paired statistical test
  - Final model: **XGBoost** (with Platt probability calibration)
- **Post-processing**:
  - Model wrapper to enforce ≈5% FPR threshold
  - Special handling for `device_fraud_count > 0`
- **Model performance on test sample:**
  - ROC-AUC: 0.893
  - Recall@5% FPR: 0.555

---

## Running the Project

### Create an .env file

Create an .env file in the api directory for example  based on the provided api/.env.example:
```bash
cp api/.env.example api/.env
```

The file contains database connection settings (for example):
```bash
DB_HOST=host.docker.internal
DB_PORT=5432
DB_NAME=fraud_detection
DB_PASSWORD=postgres
DB_USER=postgres
```

### Build the image
```bash
docker build -t fraud_detection-api:latest .
```
### Run the container
```bash
docker run -p 8000:8000 \
--add-host=host.docker.internal:host-gateway \
--env-file api/.env \
fraud_detection-api:latest
```
### Connect to the app
The API will be available at:
```
http://localhost:8000/docs
```

---

## API Endpoints

### **Home**
```
GET /
```
Redirects to Swagger UI documentation.

---

### **Check by Application ID**
```
GET /check_fraud_id
```
**Params**:
- `application_id` — List of IDs from the bank database
- `return_proba` — Whether to return prediction probabilities

---

### **Check by JSON data**
```
POST /check_fraud_json
```
**Params**:
- `data` — JSON with application features (same format as `pd.to_json(orient='records')`)
- `return_proba` — Whether to return prediction probabilities

---

## Example Requests & Responses

### Example 1 — Check by Application ID
**Request:**
```
GET http://0.0.0.0:8000/check_fraud_id?application_id=6&application_id=8&return_proba=True
```
**Response:**
```json
{
    "6": {
        "fraud_indicator": 0,
        "fraud_probability": 0.0015127136139199138
    },
    "8": {
        "fraud_indicator": 1,
        "fraud_probability": 0.06025437265634537
    }
}
```

---

### Example 2 — Check by JSON data
**Request:**
```
POST http://0.0.0.0:8000/check_fraud_json?return_proba=True
```
**Body:**  
(shortened for brevity)
```json
[
    {"income":0.5,"name_email_similarity":0.48,...},
    {"income":0.4,"name_email_similarity":0.52,...},
    {"income":0.2,"name_email_similarity":0.69,...}
]
```
**Response:**
```json
{
    "0": {
        "fraud_indicator": 0,
        "fraud_probability": 0.0006512641557492316
    },
    "1": {
        "fraud_indicator": 0,
        "fraud_probability": 0.002241698792204261
    },
    "2": {
        "fraud_indicator": 0,
        "fraud_probability": 0.0007938440539874136
    }
}
```

---

## Model File
The trained model is stored in:
```
api/app/estimator/model.pkl
```
It is loaded automatically when starting the API.

---

## License
This project is distributed under the MIT License.
