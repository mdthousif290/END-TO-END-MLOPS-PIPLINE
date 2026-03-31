import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

TARGET = 'Churn'


def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42
        ))
    ])
    return pipeline


def train(data_path: str = "data/processed/clean.csv"):
    logger.info("Starting training...")

    df = pd.read_csv(data_path)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.1)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(pipeline, "model",
                                 registered_model_name="ChurnModel")

        logger.info(f"Accuracy : {acc:.4f}")
        logger.info(f"F1 Score : {f1:.4f}")
        logger.info(f"ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred))

    logger.info("Training complete! Run: mlflow ui  to see results.")


if __name__ == "__main__":
    train()
