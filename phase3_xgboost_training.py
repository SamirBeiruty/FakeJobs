#!/usr/bin/env python3
"""
Phase 3: Train and Evaluate Random Forest Classifier for Fake Job Detection

This script:
1. Loads train/test data from Phase 2
2. Trains Random Forest classifier with specified parameters
3. Evaluates model performance comprehensively
4. Creates visualizations and saves reports
5. Saves trained model for future use
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from collections import Counter
import time
import os
import seaborn as sns

def load_phase2_data():
    """
    Load train/test data from Phase 2
    """
    print("Loading Phase 2 data...")
    
    try:
        # Load train/test splits
        splits = joblib.load('model/train_test_splits.pkl')
        X_train = splits['X_train']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_test = splits['y_test']
        
        # Load TF-IDF vectorizer for feature names
        vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        
        print("✅ Data loaded successfully")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, vectorizer
        
    except Exception as e:
        print(f"❌ Failed to load Phase 2 data: {e}")
        return None, None, None, None, None

def initialize_random_forest_model():
    """
    Initialize Random Forest classifier with specified parameters
    """
    print("\nInitializing Random Forest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=200,           # Number of trees
        max_depth=6,                # Maximum tree depth
        random_state=42,            # For reproducibility
        n_jobs=-1,                  # Use all CPU cores
        class_weight='balanced'     # Handle class imbalance
    )
    
    print("Random Forest parameters:")
    print(f"  - n_estimators: {model.n_estimators}")
    print(f"  - max_depth: {model.max_depth}")
    print(f"  - random_state: {model.random_state}")
    print(f"  - class_weight: {model.class_weight}")
    
    return model

def train_model(model, X_train, y_train):
    """
    Train Random Forest model and measure training time
    """
    print("\nTraining Random Forest model...")
    
    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    print(f"  - Training samples: {X_train.shape[0]:,}")
    print(f"  - Features: {X_train.shape[1]:,}")
    print(f"  - Training rate: {X_train.shape[0]/training_time:.0f} samples/second")
    
    return model, training_time

def make_predictions(model, X_test):
    """
    Make predictions on test set
    """
    print("\nMaking predictions on test set...")
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (fake)
    
    print(f"✅ Predictions completed")
    print(f"  - Test samples: {X_test.shape[0]:,}")
    print(f"  - Predictions shape: {y_pred.shape}")
    print(f"  - Probabilities shape: {y_proba.shape}")
    
    return y_pred, y_proba

def calculate_metrics(y_test, y_pred, y_proba):
    """
    Calculate comprehensive evaluation metrics
    """
    print("\nCalculating evaluation metrics...")
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("Model Performance Metrics:")
    print(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-Score: {f1_score:.4f}")
    print(f"  - ROC-AUC: {roc_auc:.4f}")
    
    # Check performance thresholds
    if accuracy < 0.80:
        raise ValueError(f"Model underperforming. Accuracy {accuracy:.4f} < 0.80. Consider tuning or more data.")
    
    if roc_auc < 0.80:
        raise ValueError(f"Model underperforming. ROC-AUC {roc_auc:.4f} < 0.80. Consider tuning or more data.")
    
    print("✅ Model meets performance thresholds")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc
    }

def create_visualizations(y_test, y_pred, y_proba):
    """
    Create and save visualization plots
    """
    print("\nCreating visualizations...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. ROC Curve
    print("  Creating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve
    print("  Creating Precision-Recall curve...")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion Matrix
    print("  Creating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ All visualizations saved to plots/ directory")

def save_model_and_reports(model, y_test, y_pred, metrics, vectorizer):
    """
    Save model and generate reports
    """
    print("\nSaving model and reports...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 1. Save Random Forest model
    model_path = 'models/random_forest_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # 2. Save classification report
    report_path = 'reports/classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("FAKE JOB DETECTION - CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Test set distribution
        test_counts = Counter(y_test)
        f.write(f"TEST SET DISTRIBUTION:\n")
        f.write(f"  Real jobs (0): {test_counts[0]:,} ({test_counts[0]/len(y_test)*100:.1f}%)\n")
        f.write(f"  Fake jobs (1): {test_counts[1]:,} ({test_counts[1]/len(y_test)*100:.1f}%)\n")
        f.write(f"  Total: {len(y_test):,}\n\n")
        
        # Performance metrics
        f.write(f"PERFORMANCE METRICS:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"  ROC-AUC: {metrics['roc_auc']:.4f}\n\n")
        
        # Detailed classification report
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 30 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        f.write(f"\nCONFUSION MATRIX:\n")
        f.write(f"  [[{cm[0,0]:4d} {cm[0,1]:4d}]\n")
        f.write(f"   [{cm[1,0]:4d} {cm[1,1]:4d}]]\n")
        f.write(f"  (Real predicted as Real: {cm[0,0]}, Real predicted as Fake: {cm[0,1]})\n")
        f.write(f"  (Fake predicted as Real: {cm[1,0]}, Fake predicted as Fake: {cm[1,1]})\n")
    
    print(f"✅ Classification report saved to: {report_path}")
    
    # 3. Print top feature importances
    print("\nTop 20 Most Informative Features:")
    print("-" * 40)
    
    feature_names = vectorizer.get_feature_names_out()
    feature_importances = model.feature_importances_
    
    # Get top 20 features
    top_indices = np.argsort(feature_importances)[-20:][::-1]
    
    for i, idx in enumerate(top_indices):
        importance = feature_importances[idx]
        feature_name = feature_names[idx]
        print(f"{i+1:2d}. '{feature_name}': {importance:.4f}")
    
    return model_path, report_path

def main():
    """
    Main function to execute Phase 3
    """
    print("PHASE 3: Train and Evaluate Random Forest Classifier")
    print("="*60)
    
    # Step 1: Load Phase 2 data
    X_train, X_test, y_train, y_test, vectorizer = load_phase2_data()
    if X_train is None:
        return None
    
    # Step 2: Initialize Random Forest model
    model = initialize_random_forest_model()
    
    # Step 3: Train model
    model, training_time = train_model(model, X_train, y_train)
    
    # Step 4: Make predictions
    y_pred, y_proba = make_predictions(model, X_test)
    
    # Step 5: Calculate metrics
    try:
        metrics = calculate_metrics(y_test, y_pred, y_proba)
    except ValueError as e:
        print(f"❌ {e}")
        return None
    
    # Step 6: Create visualizations
    create_visualizations(y_test, y_pred, y_proba)
    
    # Step 7: Save model and reports
    model_path, report_path = save_model_and_reports(model, y_test, y_pred, metrics, vectorizer)
    
    # Step 8: Final summary
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE - SUMMARY")
    print("="*60)
    print(f"✅ Model trained successfully")
    print(f"✅ Training time: {training_time:.2f} seconds")
    print(f"✅ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"✅ ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Report saved: {report_path}")
    print(f"✅ Visualizations saved: plots/")
    print("\n🎉 Ready for Phase 4: Cross-Validation!")
    
    return {
        'model': model,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'training_time': training_time
    }

if __name__ == "__main__":
    results = main() 