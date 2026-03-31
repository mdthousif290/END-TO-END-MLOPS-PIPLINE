import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    logger.info(f"Churn rate: {df['Churn'].mean():.2%}")
    logger.info("Data cleaning complete!")
    return df


def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data to {path}")


if __name__ == "__main__":
    raw_path = "data/raw/data.csv"
    processed_path = "data/processed/clean.csv"
    df = load_data(raw_path)
    df = clean_data(df)
    save_data(df, processed_path)
    print("Ingestion complete!")
