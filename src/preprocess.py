import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_PATH = Path("data/raw/data.csv")
PROCESSED_DIR = Path("data/processed")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def preprocess_df(df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnTransformer]:
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0.0)

    if "Churn" not in df.columns:
        raise ValueError("Raw dataset must contain a 'Churn' column.")

    target = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    df = df.drop(columns=["Churn"])

    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in df.columns if col not in numeric_features]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    features = preprocessor.fit_transform(df)

    categorical_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )
    feature_names = numeric_features + categorical_names
    processed_df = pd.DataFrame(features, columns=feature_names, index=df.index)
    processed_df["target"] = target

    return processed_df, preprocessor


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    processed_df, preprocessor = preprocess_df(df)

    processed_path = PROCESSED_DIR / "data.csv"
    processed_df.to_csv(processed_path, index=False)

    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREPROCESSOR_PATH, "wb") as handle:
        pickle.dump(preprocessor, handle)

    print(f"Created processed dataset at: {processed_path}")
    print(f"Saved preprocessing pipeline at: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
