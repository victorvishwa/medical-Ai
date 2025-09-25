#!/usr/bin/env python3
"""
Diabetes Prediction Model
Predicts whether a patient has diabetes based on clinical data like glucose levels, BMI, and blood pressure.
Uses the diabetes dataset from scikit-learn.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self):
        """Load and prepare the diabetes dataset"""
        print("Loading diabetes dataset...")
        
        # Load the diabetes dataset
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target
        
        # Create feature names
        self.feature_names = diabetes.feature_names
        
        # Convert to DataFrame for better visualization
        df = pd.DataFrame(X, columns=self.feature_names)
        df['target'] = y
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {self.feature_names}")
        print(f"Target distribution:\n{df['target'].value_counts()}")
        
        # For diabetes prediction, we'll create a binary classification problem
        # We'll use a threshold to convert regression target to classification
        # Using median as threshold (you can adjust this)
        threshold = np.median(y)
        y_binary = (y > threshold).astype(int)
        
        print(f"\nBinary classification threshold: {threshold:.2f}")
        print(f"Positive cases (diabetes): {np.sum(y_binary)}")
        print(f"Negative cases (no diabetes): {np.sum(1 - y_binary)}")
        
        return X, y_binary
    
    def split_and_scale_data(self, X, y):
        """Split data into train/test sets and scale features"""
        print("\nSplitting and scaling data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
    def train_models(self):
        """Train multiple models and select the best one"""
        print("\nTraining multiple models...")
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for Logistic Regression and SVM
            if name in ['Logistic Regression', 'SVM']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                # Use original data for Random Forest
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        # Select best model based on AUC score
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        self.model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best AUC score: {results[best_model_name]['auc']:.4f}")
        
        return results
    
    def evaluate_model(self, results):
        """Evaluate the best model and create visualizations"""
        print(f"\nEvaluating {self.best_model_name}...")
        
        best_results = results[self.best_model_name]
        y_pred = best_results['predictions']
        y_pred_proba = best_results['probabilities']
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Feature importance (for Random Forest)
        if self.best_model_name == 'Random Forest':
            feature_importance = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nFeature Importance:")
            print(importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Feature Importance for Diabetes Prediction')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{self.best_model_name} (AUC = {best_results["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Diabetes Prediction')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix - Diabetes Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_new_patient(self, patient_data):
        """Predict diabetes for a new patient"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Convert to numpy array if needed
        if isinstance(patient_data, list):
            patient_data = np.array(patient_data).reshape(1, -1)
        elif isinstance(patient_data, dict):
            # Convert dictionary to array in the correct order
            patient_data = np.array([patient_data[feature] for feature in self.feature_names]).reshape(1, -1)
        
        # Scale the data if using scaled models
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            patient_data_scaled = self.scaler.transform(patient_data)
            prediction = self.model.predict(patient_data_scaled)[0]
            probability = self.model.predict_proba(patient_data_scaled)[0][1]
        else:
            prediction = self.model.predict(patient_data)[0]
            probability = self.model.predict_proba(patient_data)[0][1]
        
        return prediction, probability
    
    def run_complete_pipeline(self):
        """Run the complete diabetes prediction pipeline"""
        print("=" * 60)
        print("DIABETES PREDICTION MODEL")
        print("=" * 60)
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Split and scale data
        self.split_and_scale_data(X, y)
        
        # Train models
        results = self.train_models()
        
        # Evaluate model
        self.evaluate_model(results)
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED")
        print("=" * 60)
        
        return results

def main():
    """Main function to run the diabetes prediction model"""
    
    # Create predictor instance
    predictor = DiabetesPredictor()
    
    # Run complete pipeline
    results = predictor.run_complete_pipeline()
    
    # Example predictions for new patients
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Example patient data (you can modify these values)
    example_patients = [
        {
            'age': 0.1, 'sex': 0.0, 'bmi': 0.1, 'bp': 0.1, 's1': 0.1, 
            's2': 0.1, 's3': 0.1, 's4': 0.1, 's5': 0.1, 's6': 0.1
        },
        {
            'age': 0.5, 'sex': 1.0, 'bmi': 0.8, 'bp': 0.7, 's1': 0.6, 
            's2': 0.5, 's3': 0.4, 's4': 0.3, 's5': 0.2, 's6': 0.1
        }
    ]
    
    for i, patient in enumerate(example_patients, 1):
        try:
            prediction, probability = predictor.predict_new_patient(patient)
            result = "Diabetes" if prediction == 1 else "No Diabetes"
            print(f"\nPatient {i}:")
            print(f"  Prediction: {result}")
            print(f"  Probability: {probability:.4f}")
            print(f"  Clinical Data: {patient}")
        except Exception as e:
            print(f"Error predicting for patient {i}: {e}")

if __name__ == "__main__":
    main()
