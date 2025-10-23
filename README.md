# Bank Account Fraud Detection API

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
│   │   ├── config
│   │   │   └── conf.py           # Getting keys from .env
│   │   ├── database              # Creating PostgresDB with SQLAlchemy
│   │   │   ├── __init__.py 
│   │   │   ├── db.py             # Creating a connection to the database
│   │   │   ├── models.py         # Describing DB tables
│   │   │   └── requests.py       # Describing DB requests
│   │   ├── estimator
│   │   │   └── model.pkl         # Final trained model
│   │   ├── __init__.py
│   │   └── schemas.py            # Describing pydantic schemas for data validation
│   ├── __init__.py
│   ├── application.py            # FastAPI app entry point 
│   └── logger_config.py          # Loguru configuration
├── data
│   ├── Bank Account Fraud Dataset
│   │   └── account_fraud.csv     # Original dataset
│   └── EDA
│       └── Fraud_EDA.ipynb       # Exploratory Data Analysis Notebook
├── training
│   ├── optuna
│   │   └── optuna_study.db        # Optuna study database for hyperparameter tuning results
│   ├── __init__.py
│   ├── dataleakage.ipynb          # Experiment with data leakage (not for production)
│   ├── DataTransformer.py         # Data preprocessing classes
│   ├── save_final_model.py        # Script for saving the model as a pickle file
│   └── training.ipynb             # Model training & tuning
├── .dockerignore
├── .env.example                   # example .env file
├── .gitattributes
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```

---

## EDA & Feature Engineering Summary
*See `data/EDA/Fraud_EDA.ipynb` for full details*
- **Train/test split**: 
  - Months 6–7 used for testing (80/20 split)
- **Target imbalance**: 
  - Fraud cases account for approximately 1% of data
- **Removed constant/irrelevant features**:
  - `device_fraud_count` - handled outside the model (if > 0, classify as fraud)
  - `zip_count_4w`, `source`, `bank_months_count`, `minutes_since_request`
- **Feature transformations**:
  - `proposed_credit_limit` converted to categorical (3 bins)
  - Rare categories merged into `Other`
- **Feature selection**: 
  - Based on correlation analysis, mutual information, feature importance scores

---

## Model Training Summary
*See `training/training.ipynb` for full details*
- **Main metric**: Recall@5% FPR (Recall at a fixed 5% False Positive Rate)
  (maximize fraud detection while keeping false positives ≤ 5%)
- **Approach**:
  - Built a reproducible end-to-end ML pipeline for data preprocessing, 
    feature engineering, and model training
  - Trained baseline models (Logistic Regression, LinearSVC)
  - Trained advanced models (RandomForest, XGBoost)
  - Applied Time Series cross-validation split by feature `Month`
  - Used SMOTE and NearMiss to rebalance the training set and increase the share of positive samples
  - Tuned hyperparameters with optuna
  - Selected the two best models and performed a paired statistical test for model selection
  - Chose the final model: **XGBoost** (with Platt scaling for probability calibration)
- **Post-processing**:
  - Implemented a model wrapper to enforce ≈5% FPR threshold
  - Added a special handling for records with `device_fraud_count > 0`
- **Model performance (test set):**
  - Recall@5% FPR: 0.545
  - PR-AUC: 0.192
  - ROC-AUC: 0.894
  - *Note*: These results are competitive. Higher scores reported in some Kaggle notebooks are often due to data leakage
  (the test set was evaluated after applying rebalancing). See `training/dataleakage.ipynb` for a demonstration.

---

## API Development Summary
*To provide access to the ML model, a simple API was implemented. See files in the `api` folder for full details*

- Architecture:
  - FastAPI backend with async support for high-performance requests
  - PostgreSQL database with SQLAlchemy ORM for storing structured application data
  - For demonstration purposes, the database is automatically initialized and optionally filled from a CSV file. 
  Data is stored in normalized tables: `Customer`, `Address`, `Application`, `Device`, `ApplicationMetrics`
  - Tables are created via SQLAlchemy model definitions
  - Asynchronous database session management is handled using `async_sessionmaker`

- Implemented Endpoints:
  - `GET /`: Redirects to API documentation (generated with FastAPI)
  - `GET /check_fraud_id`: Checks fraud by application ID from the database. Data is fetched via SQLAlchemy
  - `POST /check_fraud_json`: Checks fraud using JSON input. Input data is validated using Pydantic models

- Predictions:
  - The ML model is loaded from a pickle file when the API starts
  - The system checks whether the model supports the `predict_proba` method
  - Supports both class prediction and probability prediction (if `predict_proba` is available)
  - Prediction results are returned in the format `{application_id: {fraud_indicator, fraud_probability}}`
  - Multiple applications can be checked at once, either by ID or via JSON input
  - Input and output validation is handled via Pydantic models

- Logging:
  - Uses `loguru` for logging startup events, database operations, and prediction requests

---

## Running the Project

### Create an .env file

Create an .env file in the project directory for example based on the provided .env.example file:
```bash
cp .env.example .env
```

The file contains database connection settings (for example):
```bash
DB_HOST=db
DB_PORT=5432
DB_NAME=fraud_detection
DB_USER=postgres
DB_PASSWORD=postgres
```

### Build and run the containers
```bash
docker compose up --build
```
This command will:
- create a PostgreSQL container (db)
- create the API container (api)
- automatically create the database on the first run

To stop the containers:
```bash
docker compose down
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
        "fraud_probability": 0.0014499817043542862
    },
    "8": {
        "fraud_indicator": 1,
        "fraud_probability": 0.061979230493307114
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
