from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse

from api.app.database.db import ensure_database_exists, init_db, async_session
from api.app.database.requests import get_application_data, load_data_from_csv
from api.app.schemas import ApplicationData, Response
from api.logger_config import log

ESTIMATOR_PATH = Path.cwd() / "api" / "app" / "estimator" / "model.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    existed = await ensure_database_exists()
    await init_db()
    if not existed:
        df = pd.read_csv('data/Bank Account Fraud Dataset/account_fraud.csv')[-1000:]
        async with async_session() as session:
            await load_data_from_csv(df, session)
        log.info("Database filled")
    yield


app = FastAPI(
    swagger_ui_parameters={"syntaxHighlight": {"theme": "github"}},
    title="Fraud detector",
    lifespan=lifespan,
    description="Tool for detecting fraudulent bank applications",
)


def get_estimator(path=ESTIMATOR_PATH):
    # to save model use: joblib.dump(model, 'estimator/model.pkl')
    log.info("Estimator received")
    return joblib.load(path)


estimator = get_estimator()


def predict(data: pd.DataFrame, return_proba: bool, estimator=estimator):
    fraud_indicators = estimator.predict(data)

    fraud_probs = None
    if return_proba:
        try:
            fraud_probs = estimator.predict_proba(data)[:, 1]
        except AttributeError:
            raise HTTPException(404, "Current estimator does not support predict_proba.")

    results = {}
    for app_id, indicator in zip(data.index, fraud_indicators):
        row_result = {"fraud_indicator": int(indicator)}  # int на случай numpy.int64
        if fraud_probs is not None:
            row_result["fraud_probability"] = float(fraud_probs[data.index.get_loc(app_id)])
        results[app_id] = row_result
    return results


@app.get("/", summary="Home page redirects to Docs")
def root():
    """
    Redirect to Docs
    """
    return RedirectResponse("/docs")


@app.get("/check_fraud_id",
         response_model=Dict[int, Response],
         summary="Check for fraud by application id (from bank database)")
async def check_fraud_id(application_id: List[int] = Query(...), return_proba: bool = True):
    """
    Get application id, extract data from bank database and check for fraud
    :param application_id: application id from bank database
    :param return_proba: to get probabilities (raise an error if the estimator doesn't support probabilities prediction)
    :return: Fraud indicator and fraud probability (optional)
    """

    log.info(f"ID to check: {application_id}")
    df = await get_application_data(application_id)  # сразу одним запросом
    result = predict(df, return_proba=return_proba)
    return result


@app.post("/check_fraud_json",
          response_model=Dict[int, Response],
          summary="Check for fraud by raw data (JSON)")
def check_fraud_json(data: List[ApplicationData], return_proba: bool = True):
    """
    :param data: JSON in format like pd.to_json(orient='records')
    :param return_proba: to get probabilities (raise an error if the estimator doesn't support probabilities prediction)
    :return: Fraud indicator and fraud probability (optional)
    """
    df = pd.DataFrame([application.model_dump() for application in data])
    result = predict(df, return_proba=return_proba)
    return result