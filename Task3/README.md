# Task 3: Logistic Regression - Heart Disease UCI

This folder trains, evaluates, and interprets a Logistic Regression model on the Heart Disease UCI dataset.

## Files
- `heart_statlog_cleveland_hungary_final.csv`: dataset used (expects column `target`).
- `logistic_regression_heart.py`: end-to-end pipeline (understanding, split+scale, train, evaluate, save artifacts).
- `requirements.txt`: Python dependencies.

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python logistic_regression_heart.py
```

## Outputs
- `logreg_model.joblib`: trained Logistic Regression model
- `scaler.joblib`: fitted `StandardScaler`
- `logreg_confusion_matrix.png`: confusion matrix plot
- `logreg_roc_curve.png`: ROC curve plot
- `logreg_classification_report.txt`: precision/recall/F1 per class (text)
- `logreg_metrics.json`: metrics summary as JSON
- `data_understanding.json`: dataset summary (rows, columns, missing, class distribution)

## Notes
- Uses an 80/20 train/test split with stratification and standard scaling.
- Threshold 0.5 for class prediction; AUC computed from probabilities.
