# End-to-End MLOps Pipeline

This project is a starter MLOps pipeline built with:
- Python 3.10+
- FastAPI
- MLflow
- DVC
- Evidently
- scikit-learn
- XGBoost

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git init
.\venv\Scripts\dvc.exe init
```

## Project layout

- `data/raw` - raw ingested data
- `data/processed` - cleaned / feature-engineered dataset
- `models` - trained model and metrics
- `src` - pipeline scripts
- `notebooks` - optional exploratory notebooks

## Run pipeline stages

```powershell
python src\ingest.py
python src\preprocess.py
python src\train.py
python src\evaluate.py
python src\api.py
```

## API

Start the FastAPI app with:

```powershell
uvicorn src.api:app --reload
```
