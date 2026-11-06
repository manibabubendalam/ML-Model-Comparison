"""Train the five classifiers on the UCI Dry Bean dataset and save them to model/.

Usage:
    python train_models.py

Outputs:
    data/dry_bean.csv          full dataset (converted from the UCI .xlsx)
    test_data.csv              20% stratified hold-out set (features + Class) for the app
    model/<name>.joblib        one fitted sklearn Pipeline per classifier
    model/metrics.csv          evaluation metrics of every model on test_data.csv
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from model_utils import MODEL_FILES, build_models, evaluate

ROOT = Path(__file__).parent
RAW_XLSX = ROOT / "data" / "raw" / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx"
FULL_CSV = ROOT / "data" / "dry_bean.csv"
TEST_CSV = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Class"
SEED = 42


def load_dataset() -> pd.DataFrame:
    if not FULL_CSV.exists():
        pd.read_excel(RAW_XLSX).to_csv(FULL_CSV, index=False)
    return pd.read_csv(FULL_CSV)


def main() -> None:
    df = load_dataset()
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1] - 1} features, {df[TARGET].nunique()} classes")

    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_CSV, index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    rows = []
    for name, pipeline in build_models(SEED).items():
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODEL_DIR / MODEL_FILES[name])
        scores = evaluate(pipeline, X_test, y_test)
        rows.append({"ML Model Name": name, **scores})
        print(f"{name:<22}", {k: round(v, 4) for k, v in scores.items()})

    pd.DataFrame(rows).to_csv(MODEL_DIR / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
