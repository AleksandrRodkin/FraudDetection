import joblib
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from training.DataTransformer import NoneHandler, DataHandler, DataPreprocessor, IsoForest, ClassifierWrapper

df = pd.read_csv(PROJECT_ROOT / "data" / "Bank Account Fraud Dataset" / "account_fraud.csv")

X_train, y_train = df[~df.month.isin([6, 7])].drop(['fraud_bool'], axis=1), df[~df.month.isin([6, 7])]['fraud_bool']
X_test, y_test = df[df.month.isin([6, 7])].drop(['fraud_bool'], axis=1), df[df.month.isin([6, 7])]['fraud_bool']

custom_funcs = (
    ('similarity', ('device_os', ['customer_age', 'income'])),
    ('similarity', ('housing_status', ['customer_age', 'income'])),
)


dp = DataPreprocessor(
    cat_features=[
        'email_is_free', 'has_other_cards',  'keep_alive_session', 'phone_home_valid',  'phone_mobile_valid', 'device_os',
        'employment_status', 'customer_age', 'payment_type',  'housing_status', 'foreign_request',  'income', 'device_distinct_emails_8w',
        'proposed_credit_limit',
    ],
    num_features=[
        'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 'intended_balcon_amount',
        'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'session_length_in_minutes', 'days_since_request',
        'velocity_6h', 'velocity_24h', 'velocity_4w',
    ],
    custom_functions=custom_funcs,
    num_ohe_mte_threshold=5
)


preprocessor = Pipeline([
    ('none_handler', NoneHandler()),
    ('handler', DataHandler()),
    ('data_preprocessor', dp),
    ('iso', IsoForest(contamination='auto')),
])

final_xgb_params = {
    "booster":'gbtree',
    "scale_pos_weight": (len(y_train) - np.sum(y_train)) / np.sum(y_train),
    "n_estimators": 650,
    "max_depth": 2,
    "learning_rate": 0.12,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 0.26,
    "gamma": 0.13,
    "reg_alpha": 1.5e-4,
    "reg_lambda": 80,
    "eval_metric": "logloss",
    "seed": 42
}


final_xgb = XGBClassifier(**final_xgb_params)

final_pipe_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', final_xgb),
])

platts_calibration = CalibratedClassifierCV(
    final_pipe_xgb,
    method='sigmoid',
    ensemble=False,
    cv=5,
    n_jobs = -1
)

wrapped_classifier = ClassifierWrapper(platts_calibration).fit(X_train, y_train)

ESTIMATOR_PATH = PROJECT_ROOT / 'api' / 'app' / "estimator" / "model.pkl"

joblib.dump(wrapped_classifier, ESTIMATOR_PATH)