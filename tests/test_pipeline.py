import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest import clean_data

def make_sample_df():
    return pd.DataFrame({
        'customerID': ['1234-ABCD'],
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [12],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['Yes'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [70.35],
        'TotalCharges': ['845.5'],
        'Churn': ['No']
    })

def test_clean_data_removes_customerID():
    df = make_sample_df()
    cleaned = clean_data(df)
    assert 'customerID' not in cleaned.columns

def test_clean_data_churn_is_binary():
    df = make_sample_df()
    cleaned = clean_data(df)
    assert set(cleaned['Churn'].unique()).issubset({0, 1})

def test_clean_data_total_charges_numeric():
    df = make_sample_df()
    cleaned = clean_data(df)
    assert pd.api.types.is_numeric_dtype(cleaned['TotalCharges'])

def test_clean_data_no_nulls():
    df = make_sample_df()
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0
