import os
import json
import joblib
import numpy as np
import pandas as pd

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "heart_statlog_cleveland_hungary_final.csv"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def clean_and_prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Missing target column '{TARGET_COLUMN}'")

    # Remove rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # Select features (all except target) and coerce to numeric where possible
    X = df.drop(columns=[TARGET_COLUMN]).apply(pd.to_numeric, errors="coerce")
    y = df[TARGET_COLUMN].astype(int).values

    # Impute missing values with median by default
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    return pd.DataFrame(X_imputed, columns=X.columns), y


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, DATA_FILE)

    df = load_data(data_path)
    X, y = clean_and_prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = train_random_forest(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=3))

    # Print a sample prediction for the first test row
    sample_features = X_test.iloc[0].to_numpy().reshape(1, -1)
    sample_pred = clf.predict(sample_features)[0]
    sample_proba = clf.predict_proba(sample_features)[0, 1]
    print("\nSample prediction on first test instance:")
    print(f"Predicted class: {int(sample_pred)} | Probability of class 1: {sample_proba:.3f}")

    # Save model and a small metrics JSON
    joblib.dump(clf, os.path.join(cwd, "random_forest_model.joblib"))
    with open(os.path.join(cwd, "rf_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": float(acc)}, f, indent=2)


if __name__ == "__main__":
    main()
