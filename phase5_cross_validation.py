#!/usr/bin/env python3
"""
Phase 5: 5-fold Cross-Validation and Hyperparameter Tuning

This script:
1. Loads training data from previous phases
2. Performs 5-fold stratified cross-validation
3. Uses RandomizedSearchCV for hyperparameter tuning
4. Saves the best model and parameters
5. Provides comprehensive evaluation results
"""

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import time
import os
from collections import Counter

def load_training_data():
    """
    Load training data from Phase 2
    """
    print("Loading training data...")
    
    try:
        # Load train/test splits
        splits = joblib.load('model/train_test_splits.pkl')
        X_train = splits['X_train']
        y_train = splits['y_train']
        
        print("✅ Training data loaded successfully")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        
        # Print class distribution
        train_counts = Counter(y_train)
        print(f"  Class distribution:")
        print(f"    Real jobs (0): {train_counts[0]:,} ({train_counts[0]/len(y_train)*100:.1f}%)")
        print(f"    Fake jobs (1): {train_counts[1]:,} ({train_counts[1]/len(y_train)*100:.1f}%)")
        
        return X_train, y_train
        
    except Exception as e:
        print(f"❌ Failed to load training data: {e}")
        return None, None

def define_parameter_grid():
    """
    Define parameter grid for RandomizedSearchCV
    """
    print("\nDefining parameter grid...")
    
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [4, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    print("✅ Parameter grid defined:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"  Total possible combinations: {total_combinations}")
    print(f"  Will test {30} combinations (n_iter=30)")
    
    return param_grid

def setup_cross_validation():
    """
    Setup cross-validation strategy
    """
    print("\nSetting up cross-validation...")
    
    # Create stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("✅ Cross-validation setup:")
    print(f"  Strategy: StratifiedKFold")
    print(f"  n_splits: 5")
    print(f"  shuffle: True")
    print(f"  random_state: 42")
    
    return cv

def create_base_model():
    """
    Create base Random Forest model
    """
    print("\nCreating base Random Forest model...")
    
    base_model = RandomForestClassifier(
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    print("✅ Base model created:")
    print(f"  Algorithm: RandomForestClassifier")
    print(f"  class_weight: balanced")
    print(f"  random_state: 42")
    print(f"  n_jobs: -1 (all CPU cores)")
    
    return base_model

def perform_hyperparameter_tuning(X_train, y_train, base_model, param_grid, cv):
    """
    Perform hyperparameter tuning using RandomizedSearchCV
    """
    print("\nPerforming hyperparameter tuning...")
    print("="*60)
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=30,                    # Number of parameter combinations to try
        scoring='f1',                 # Optimize for F1 score
        cv=cv,                        # 5-fold stratified cross-validation
        verbose=2,                    # Detailed output
        n_jobs=-1,                    # Use all CPU cores
        random_state=42,              # For reproducibility
        return_train_score=True       # Also return training scores
    )
    
    print("RandomizedSearchCV parameters:")
    print(f"  n_iter: 30")
    print(f"  scoring: f1")
    print(f"  cv: 5-fold stratified")
    print(f"  verbose: 2")
    print(f"  n_jobs: -1")
    print(f"  random_state: 42")
    
    # Fit the search
    print("\nStarting hyperparameter search...")
    start_time = time.time()
    
    random_search.fit(X_train, y_train)
    
    end_time = time.time()
    search_time = end_time - start_time
    
    print(f"\n✅ Hyperparameter search completed in {search_time:.2f} seconds")
    
    return random_search, search_time

def analyze_results(random_search, search_time):
    """
    Analyze and print cross-validation results
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    # Best parameters
    print("\n🏆 BEST HYPERPARAMETERS:")
    print("-" * 30)
    best_params = random_search.best_params_
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Best score
    print(f"\n📊 BEST MODEL PERFORMANCE:")
    print("-" * 30)
    print(f"  Best F1 Score: {random_search.best_score_:.4f}")
    print(f"  Best Score Std: {random_search.cv_results_['std_test_score'][random_search.best_index_]:.4f}")
    
    # Cross-validation results summary
    print(f"\n📈 CROSS-VALIDATION SUMMARY:")
    print("-" * 30)
    cv_results = random_search.cv_results_
    
    # Get scores for all folds of the best model
    best_index = random_search.best_index_
    fold_scores = []
    for i in range(5):  # 5 folds
        fold_score = cv_results[f'split{i}_test_score'][best_index]
        fold_scores.append(fold_score)
    
    print(f"  F1 Scores across 5 folds:")
    for i, score in enumerate(fold_scores, 1):
        print(f"    Fold {i}: {score:.4f}")
    
    print(f"  Mean F1 Score: {np.mean(fold_scores):.4f}")
    print(f"  Std F1 Score: {np.std(fold_scores):.4f}")
    print(f"  Min F1 Score: {np.min(fold_scores):.4f}")
    print(f"  Max F1 Score: {np.max(fold_scores):.4f}")
    
    # Search statistics
    print(f"\n🔍 SEARCH STATISTICS:")
    print("-" * 30)
    print(f"  Total combinations tested: {len(cv_results['mean_test_score'])}")
    print(f"  Search time: {search_time:.2f} seconds")
    print(f"  Average time per combination: {search_time/len(cv_results['mean_test_score']):.2f} seconds")
    
    # Top 5 parameter combinations
    print(f"\n🏅 TOP 5 PARAMETER COMBINATIONS:")
    print("-" * 30)
    
    # Sort by test score
    sorted_indices = np.argsort(cv_results['mean_test_score'])[::-1]
    
    for i, idx in enumerate(sorted_indices[:5], 1):
        score = cv_results['mean_test_score'][idx]
        std = cv_results['std_test_score'][idx]
        
        print(f"\n  {i}. F1 Score: {score:.4f} (±{std:.4f})")
        for param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'bootstrap']:
            param_key = f'param_{param}'
            if param_key in cv_results:
                value = cv_results[param_key][idx]
                print(f"     {param}: {value}")

def save_best_model(random_search):
    """
    Save the best model and parameters
    """
    print("\nSaving best model and parameters...")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Save best model
    best_model = random_search.best_estimator_
    model_path = 'models/random_forest_cv.pkl'
    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved to: {model_path}")
    
    # Save best parameters as JSON
    best_params = random_search.best_params_
    params_path = 'reports/rf_cv_best_params.json'
    
    # Convert numpy types to native Python types for JSON serialization
    json_params = {}
    for key, value in best_params.items():
        if value is None:
            json_params[key] = None
        elif isinstance(value, (np.integer, np.floating)):
            json_params[key] = value.item()
        else:
            json_params[key] = value
    
    with open(params_path, 'w') as f:
        json.dump(json_params, f, indent=2)
    
    print(f"✅ Best parameters saved to: {params_path}")
    
    return model_path, params_path

def compare_with_original_model(best_model, X_train, y_train):
    """
    Compare the tuned model with the original model
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    try:
        # Load original model
        original_model = joblib.load('models/random_forest_model.pkl')
        
        # Load test data
        splits = joblib.load('model/train_test_splits.pkl')
        X_test = splits['X_test']
        y_test = splits['y_test']
        
        print("\n📊 PERFORMANCE COMPARISON ON TEST SET:")
        print("-" * 40)
        
        # Test original model
        y_pred_orig = original_model.predict(X_test)
        y_proba_orig = original_model.predict_proba(X_test)[:, 1]
        
        # Test tuned model
        y_pred_tuned = best_model.predict(X_test)
        y_proba_tuned = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics_orig = {
            'accuracy': accuracy_score(y_test, y_pred_orig),
            'precision': precision_score(y_test, y_pred_orig),
            'recall': recall_score(y_test, y_pred_orig),
            'f1': f1_score(y_test, y_pred_orig),
            'roc_auc': roc_auc_score(y_test, y_proba_orig)
        }
        
        metrics_tuned = {
            'accuracy': accuracy_score(y_test, y_pred_tuned),
            'precision': precision_score(y_test, y_pred_tuned),
            'recall': recall_score(y_test, y_pred_tuned),
            'f1': f1_score(y_test, y_pred_tuned),
            'roc_auc': roc_auc_score(y_test, y_proba_tuned)
        }
        
        print(f"{'Metric':<12} {'Original':<10} {'Tuned':<10} {'Improvement':<12}")
        print("-" * 50)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            orig_val = metrics_orig[metric]
            tuned_val = metrics_tuned[metric]
            improvement = tuned_val - orig_val
            
            print(f"{metric:<12} {orig_val:<10.4f} {tuned_val:<10.4f} {improvement:>+10.4f}")
        
        # Model complexity comparison
        print(f"\n🔧 MODEL COMPLEXITY COMPARISON:")
        print("-" * 40)
        print(f"Original model parameters:")
        for param, value in original_model.get_params().items():
            if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'bootstrap']:
                print(f"  {param}: {value}")
        
        print(f"\nTuned model parameters:")
        for param, value in best_model.get_params().items():
            if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'bootstrap']:
                print(f"  {param}: {value}")
        
    except Exception as e:
        print(f"⚠️  Could not compare with original model: {e}")

def main():
    """
    Main function to execute Phase 5
    """
    print("PHASE 5: 5-fold Cross-Validation and Hyperparameter Tuning")
    print("="*70)
    
    # Step 1: Load training data
    X_train, y_train = load_training_data()
    if X_train is None:
        return None
    
    # Step 2: Define parameter grid
    param_grid = define_parameter_grid()
    
    # Step 3: Setup cross-validation
    cv = setup_cross_validation()
    
    # Step 4: Create base model
    base_model = create_base_model()
    
    # Step 5: Perform hyperparameter tuning
    random_search, search_time = perform_hyperparameter_tuning(X_train, y_train, base_model, param_grid, cv)
    
    # Step 6: Analyze results
    analyze_results(random_search, search_time)
    
    # Step 7: Save best model and parameters
    model_path, params_path = save_best_model(random_search)
    
    # Step 8: Compare with original model
    compare_with_original_model(random_search.best_estimator_, X_train, y_train)
    
    # Step 9: Final summary
    print("\n" + "="*70)
    print("PHASE 5 COMPLETE - SUMMARY")
    print("="*70)
    print("✅ 5-fold cross-validation completed")
    print("✅ Hyperparameter tuning completed")
    print(f"✅ Best F1 Score: {random_search.best_score_:.4f}")
    print(f"✅ Search time: {search_time:.2f} seconds")
    print(f"✅ Best model saved: {model_path}")
    print(f"✅ Best parameters saved: {params_path}")
    print("\n🎉 Ready for Phase 6: Model Serialization & Optimization!")
    
    return {
        'best_model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'search_time': search_time,
        'cv_results': random_search.cv_results_
    }

if __name__ == "__main__":
    results = main() 