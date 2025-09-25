import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and explore the heart disease dataset"""
    print("=" * 60)
    print("HEART DISEASE UCI DATASET - SVM WITH RBF KERNEL")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Samples: {df.shape[0]}")
    
    # Display basic information
    print("\nDataset Info:")
    print(df.info())
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Check target distribution
    print(f"\nTarget Distribution:")
    print(df['target'].value_counts())
    print(f"Target Distribution (%):")
    print(df['target'].value_counts(normalize=True) * 100)
    
    return df

def preprocess_data(df):
    """Preprocess the data for machine learning"""
    print("\n" + "=" * 40)
    print("DATA PREPROCESSING")
    print("=" * 40)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Check for any missing values in features
    if X.isnull().sum().sum() > 0:
        print("Handling missing values...")
        X = X.fillna(X.median())
    
    # Feature scaling
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("Data preprocessing completed!")
    return X_scaled, y, scaler

def split_data(X, y):
    """Split data into training and testing sets"""
    print("\n" + "=" * 40)
    print("DATA SPLITTING")
    print("=" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Training set target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test set target distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def train_svm_basic(X_train, y_train):
    """Train basic SVM with RBF kernel"""
    print("\n" + "=" * 40)
    print("BASIC SVM TRAINING")
    print("=" * 40)
    
    # Create SVM with RBF kernel
    svm_basic = SVC(kernel='rbf', random_state=42, probability=True)
    
    # Train the model
    print("Training basic SVM with RBF kernel...")
    svm_basic.fit(X_train, y_train)
    
    print("Basic SVM training completed!")
    return svm_basic

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning with GridSearchCV"""
    print("\n" + "=" * 40)
    print("HYPERPARAMETER TUNING")
    print("=" * 40)
    
    # Define parameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Create SVM model
    svm = SVC(random_state=42, probability=True)
    
    # Perform GridSearchCV
    print("\nPerforming GridSearchCV (this may take a few minutes)...")
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search

def evaluate_model(model, X_test, y_test, model_name="SVM"):
    """Evaluate the model with comprehensive metrics"""
    print(f"\n" + "=" * 40)
    print(f"{model_name.upper()} MODEL EVALUATION")
    print("=" * 40)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }

def create_visualizations(y_test, basic_results, tuned_results):
    """Create visualizations for model performance"""
    print("\n" + "=" * 40)
    print("CREATING VISUALIZATIONS")
    print("=" * 40)
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SVM with RBF Kernel - Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix Comparison
    ax1 = axes[0, 0]
    sns.heatmap(basic_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Basic SVM - Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    ax2 = axes[0, 1]
    sns.heatmap(tuned_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Tuned SVM - Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # 2. ROC Curves
    ax3 = axes[0, 2]
    fpr_basic, tpr_basic, _ = roc_curve(y_test, basic_results['y_pred_proba'])
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, tuned_results['y_pred_proba'])
    
    ax3.plot(fpr_basic, tpr_basic, label=f'Basic SVM (AUC = {basic_results["roc_auc"]:.3f})', linewidth=2)
    ax3.plot(fpr_tuned, tpr_tuned, label=f'Tuned SVM (AUC = {tuned_results["roc_auc"]:.3f})', linewidth=2)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. Metrics Comparison
    ax4 = axes[1, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    basic_scores = [basic_results['accuracy'], basic_results['precision'], 
                   basic_results['recall'], basic_results['f1'], basic_results['roc_auc']]
    tuned_scores = [tuned_results['accuracy'], tuned_results['precision'], 
                   tuned_results['recall'], tuned_results['f1'], tuned_results['roc_auc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, basic_scores, width, label='Basic SVM', alpha=0.8)
    ax4.bar(x + width/2, tuned_scores, width, label='Tuned SVM', alpha=0.8)
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Score')
    ax4.set_title('Model Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 4. Cross-validation scores (if available)
    ax5 = axes[1, 1]
    # This would be populated if we had cross-validation results
    ax5.text(0.5, 0.5, 'Cross-validation\nscores would be\nplotted here', 
             ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Cross-Validation Scores')
    ax5.axis('off')
    
    # 5. Feature importance (SVM doesn't have direct feature importance, so we'll show model parameters)
    ax6 = axes[1, 2]
    ax6.text(0.5, 0.5, 'SVM Parameters:\n• Kernel: RBF\n• C: Optimized\n• Gamma: Optimized', 
             ha='center', va='center', transform=ax6.transAxes, fontsize=12)
    ax6.set_title('Model Configuration')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('svm_heart_disease_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'svm_heart_disease_analysis.png'")

def cross_validation_analysis(model, X, y):
    """Perform cross-validation analysis"""
    print("\n" + "=" * 40)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 40)
    
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def main():
    """Main function to run the complete SVM analysis"""
    try:
        # Load and explore data
        df = load_and_explore_data()
        
        # Preprocess data
        X, y, scaler = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train basic SVM
        svm_basic = train_svm_basic(X_train, y_train)
        
        # Hyperparameter tuning
        svm_tuned, grid_search = hyperparameter_tuning(X_train, y_train)
        
        # Evaluate both models
        basic_results = evaluate_model(svm_basic, X_test, y_test, "Basic SVM")
        tuned_results = evaluate_model(svm_tuned, X_test, y_test, "Tuned SVM")
        
        # Cross-validation analysis
        cv_scores = cross_validation_analysis(svm_tuned, X, y)
        
        # Create visualizations
        create_visualizations(y_test, basic_results, tuned_results)
        
        # Final summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Best SVM Model Performance:")
        print(f"  Accuracy: {tuned_results['accuracy']:.4f}")
        print(f"  Precision: {tuned_results['precision']:.4f}")
        print(f"  Recall: {tuned_results['recall']:.4f}")
        print(f"  F1-Score: {tuned_results['f1']:.4f}")
        print(f"  ROC-AUC: {tuned_results['roc_auc']:.4f}")
        print(f"  Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print(f"\nBest hyperparameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

