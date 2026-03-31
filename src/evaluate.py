import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report


def main() -> None:
    processed_path = Path("data/processed/data.csv")
    model_path = Path("models/model.pkl")
    model_dir = Path("models")
    report_path = model_dir / "evaluation.json"

    df = pd.read_csv(processed_path)
    if "target" not in df.columns:
        raise ValueError("Processed dataset must contain a 'target' column.")

    X = df.drop(columns=["target"])
    y = df["target"]

    with open(model_path, "rb") as handle:
        model = pickle.load(handle)

    predictions = model.predict(X)
    report = classification_report(y, predictions, output_dict=True)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved evaluation report to: {report_path}")


if __name__ == "__main__":
    main()
