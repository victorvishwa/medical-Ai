import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple, List
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    classification_report,
)


DATA_FILE = "heart_statlog_cleveland_hungary_final.csv"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class DatasetInfo:
    num_rows: int
    num_columns: int
    columns: List[str]
    num_missing_values: int
    class_distribution: dict


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def understand_data(df: pd.DataFrame) -> DatasetInfo:
    num_rows, num_columns = df.shape
    num_missing_values = int(df.isna().sum().sum())
    class_distribution = (
        df[TARGET_COLUMN].value_counts().sort_index().to_dict()
        if TARGET_COLUMN in df.columns
        else {}
    )
    return DatasetInfo(
        num_rows=num_rows,
        num_columns=num_columns,
        columns=df.columns.tolist(),
        num_missing_values=num_missing_values,
        class_distribution=class_distribution,
    )


def preprocess(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Missing target column '{TARGET_COLUMN}' in dataset")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int).values

    # Identify categorical-like columns by dtype or known names and one-hot encode if any
    # In this dataset, categorical features are already numeric-encoded; proceed with scaling only

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled, y, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    # Save confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(output_dir, "logreg_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Save ROC curve plot
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve - Logistic Regression")
    roc_path = os.path.join(output_dir, "logreg_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Save classification report text
    report_path = os.path.join(output_dir, "logreg_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report_path": report_path,
        "confusion_matrix_image": cm_path,
        "roc_curve_image": roc_path,
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "logreg_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cwd, DATA_FILE)
    outputs_dir = cwd  # Save outputs to Task3 folder

    # Task 1: Data Understanding
    df = load_data(data_path)
    info = understand_data(df)

    # Save a brief data understanding JSON
    data_understanding = {
        "num_rows": info.num_rows,
        "num_columns": info.num_columns,
        "columns": info.columns,
        "num_missing_values": info.num_missing_values,
        "class_distribution": info.class_distribution,
    }
    with open(os.path.join(outputs_dir, "data_understanding.json"), "w", encoding="utf-8") as f:
        json.dump(data_understanding, f, indent=2)

    # Task 2: Train-Test Split & Feature Scaling
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Task 3: Train Logistic Regression Model
    model = train_model(X_train, y_train)

    # Persist artifacts
    joblib.dump(model, os.path.join(outputs_dir, "logreg_model.joblib"))
    joblib.dump(scaler, os.path.join(outputs_dir, "scaler.joblib"))

    # Task 4: Model Prediction & Evaluation
    metrics = evaluate_model(model, X_test, y_test, outputs_dir)

    # Print concise summary to console
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }, indent=2))


if __name__ == "__main__":
    main()
