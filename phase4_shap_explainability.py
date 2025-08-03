#!/usr/bin/env python3
"""
Phase 4: Model Explainability with SHAP for Random Forest Job Fraud Classifier

This script:
1. Loads trained model and data from previous phases
2. Prepares data for SHAP analysis
3. Computes SHAP values for model explainability
4. Creates visualizations of feature importance
5. Saves explainer for future use
"""

import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
from collections import Counter

def load_required_objects():
    """
    Load model, vectorizer, and data from previous phases
    """
    print("Loading required objects...")
    
    try:
        # Load Random Forest model
        model = joblib.load('models/random_forest_model.pkl')
        print("✅ Random Forest model loaded")
        
        # Load TF-IDF vectorizer
        vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        print("✅ TF-IDF vectorizer loaded")
        
        # Load train/test data
        splits = joblib.load('model/train_test_splits.pkl')
        X_train = splits['X_train']
        X_test = splits['X_test']
        y_test = splits['y_test']
        print("✅ Train/test data loaded")
        
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        return model, vectorizer, X_train, X_test, y_test
        
    except Exception as e:
        print(f"❌ Failed to load required objects: {e}")
        return None, None, None, None, None

def prepare_feature_names(vectorizer):
    """
    Get feature names from TF-IDF vectorizer
    """
    print("\nPreparing feature names...")
    
    feature_names = vectorizer.get_feature_names_out()
    print(f"✅ Feature names prepared: {len(feature_names)} features")
    print(f"  Sample features: {feature_names[:10].tolist()}")
    
    return feature_names

def convert_to_dense_format(X_train, max_samples=500):
    """
    Convert sparse matrix to dense format for SHAP
    """
    print(f"\nConverting to dense format...")
    print(f"  Original X_train shape: {X_train.shape}")
    print(f"  Memory estimate: {X_train.shape[0] * X_train.shape[1] * 8 / 1e6:.1f} MB")
    
    # Check if we need to subset due to memory constraints
    if X_train.shape[0] > max_samples:
        print(f"  ⚠️  Large dataset detected. Subsetting to {max_samples} samples for SHAP analysis")
        # Take a stratified sample
        from sklearn.model_selection import train_test_split
        indices = np.arange(X_train.shape[0])
        sample_indices, _ = train_test_split(
            indices, 
            train_size=max_samples, 
            random_state=42,
            stratify=None  # We'll handle stratification manually if needed
        )
        X_train_subset = X_train[sample_indices]
        print(f"  Subset shape: {X_train_subset.shape}")
    else:
        X_train_subset = X_train
        sample_indices = np.arange(X_train.shape[0])
    
    # Convert to dense format
    print("  Converting sparse matrix to dense...")
    X_train_dense = X_train_subset.toarray()
    
    print(f"✅ Dense conversion complete")
    print(f"  Dense shape: {X_train_dense.shape}")
    print(f"  Memory used: {X_train_dense.nbytes / 1e6:.1f} MB")
    
    return X_train_dense, sample_indices

def initialize_shap_explainer(model, X_train_dense):
    """
    Initialize SHAP explainer
    """
    print("\nInitializing SHAP explainer...")
    
    try:
        # Try TreeExplainer first (works well with Random Forest)
        explainer = shap.TreeExplainer(model)
        print("✅ TreeExplainer initialized successfully")
    except Exception as e:
        print(f"⚠️  TreeExplainer failed: {e}")
        print("  Falling back to Explainer...")
        try:
            explainer = shap.Explainer(model.predict, X_train_dense)
            print("✅ Explainer initialized successfully")
        except Exception as e2:
            print(f"❌ Explainer also failed: {e2}")
            return None
    
    return explainer

def compute_shap_values(explainer, X_train_dense, X_test=None):
    """
    Compute SHAP values for training data and optionally test data
    """
    print("\nComputing SHAP values...")
    
    # Compute for training data
    print("  Computing SHAP values for training data...")
    shap_values = explainer(X_train_dense)
    print(f"✅ Training SHAP values computed: {shap_values.shape}")
    
    # Optionally compute for test data
    if X_test is not None:
        print("  Computing SHAP values for test data...")
        X_test_dense = X_test.toarray()
        shap_values_test = explainer(X_test_dense)
        print(f"✅ Test SHAP values computed: {shap_values_test.shape}")
    else:
        shap_values_test = None
    
    return shap_values, shap_values_test

def create_shap_visualizations(shap_values, feature_names, X_train_dense):
    """
    Create SHAP visualizations
    """
    print("\nCreating SHAP visualizations...")
    
    # Create explainability directory
    os.makedirs('explainability', exist_ok=True)
    
    # Set matplotlib style for better plots
    plt.style.use('default')
    
    # 1. SHAP Summary Plot (Beeswarm) - Use only the positive class SHAP values
    print("  Creating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    # Use only the positive class (fraud) SHAP values for the beeswarm plot
    shap.plots.beeswarm(shap_values[:, :, 1], max_display=20, show=False)
    plt.title("SHAP Summary Plot - Feature Importance for Fake Job Detection", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('explainability/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved")
    
    # 2. Top 20 Features Bar Chart
    print("  Creating feature importance bar chart...")
    create_feature_importance_bar(shap_values, feature_names)
    print("✅ Feature importance bar chart saved")
    
    # 3. Waterfall plot for a sample prediction
    print("  Creating sample prediction waterfall plot...")
    create_sample_waterfall(shap_values, feature_names, X_train_dense)
    print("✅ Sample waterfall plot saved")

def create_feature_importance_bar(shap_values, feature_names):
    """
    Create horizontal bar chart of top 20 features by mean absolute SHAP value
    """
    # Calculate mean absolute SHAP values (use positive class)
    mean_abs_shap = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
    
    # Get top 20 features
    top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
    top_features = feature_names[top_indices]
    top_importance = mean_abs_shap[top_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(top_features)), top_importance, color='steelblue')
    
    # Customize the plot
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title('Top 20 Most Important Features for Fake Job Detection', fontsize=14, pad=20)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_importance)):
        plt.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('explainability/shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sample_waterfall(shap_values, feature_names, X_train_dense):
    """
    Create waterfall plot for a sample prediction
    """
    # Find a sample with high SHAP values for demonstration (use positive class)
    sample_idx = np.argmax(np.abs(shap_values.values[:, :, 1]).sum(axis=1))
    
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_values[sample_idx, :, 1], max_display=15, show=False)
    plt.title(f"SHAP Waterfall Plot - Sample Prediction {sample_idx}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('explainability/shap_waterfall_sample.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_explainer(explainer):
    """
    Save SHAP explainer for future use
    """
    print("\nSaving SHAP explainer...")
    
    try:
        joblib.dump(explainer, 'explainability/shap_explainer.pkl')
        print("✅ SHAP explainer saved to: explainability/shap_explainer.pkl")
        return True
    except Exception as e:
        print(f"❌ Failed to save explainer: {e}")
        return False

def analyze_feature_importance(shap_values, feature_names):
    """
    Analyze and print top positive and negative features
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Calculate mean SHAP values (positive = pushes toward fraud, negative = pushes toward real)
    # Use positive class (fraud) SHAP values
    mean_shap = shap_values.values[:, :, 1].mean(axis=0)
    
    # Get top positive features (push toward fraud)
    top_positive_indices = np.argsort(mean_shap)[-10:][::-1]
    top_positive_features = feature_names[top_positive_indices]
    top_positive_values = mean_shap[top_positive_indices]
    
    # Get top negative features (push toward real)
    top_negative_indices = np.argsort(mean_shap)[:10]
    top_negative_features = feature_names[top_negative_indices]
    top_negative_values = mean_shap[top_negative_indices]
    
    print("\n🔴 TOP 10 FEATURES PUSHING TOWARD FRAUD (Positive SHAP):")
    print("-" * 50)
    for i, (feature, value) in enumerate(zip(top_positive_features, top_positive_values), 1):
        print(f"{i:2d}. '{feature}': {value:.4f}")
    
    print("\n🟢 TOP 10 FEATURES PUSHING TOWARD REAL (Negative SHAP):")
    print("-" * 50)
    for i, (feature, value) in enumerate(zip(top_negative_features, top_negative_values), 1):
        print(f"{i:2d}. '{feature}': {value:.4f}")
    
    # Explain the most important feature
    most_important_idx = np.argmax(np.abs(mean_shap))
    most_important_feature = feature_names[most_important_idx]
    most_important_value = mean_shap[most_important_idx]
    
    print(f"\n🎯 MOST IMPORTANT FEATURE: '{most_important_feature}' (SHAP: {most_important_value:.4f})")
    print("-" * 50)
    
    if most_important_value > 0:
        print(f"   This feature PUSHES predictions toward FRAUD")
        print(f"   When '{most_important_feature}' appears in job postings, it increases")
        print(f"   the likelihood of the posting being classified as fake.")
    else:
        print(f"   This feature PUSHES predictions toward REAL")
        print(f"   When '{most_important_feature}' appears in job postings, it increases")
        print(f"   the likelihood of the posting being classified as legitimate.")
    
    # Provide interpretation for common features
    print(f"\n📊 FEATURE INTERPRETATION:")
    print("-" * 30)
    interpret_features(top_positive_features[:5], "fraud indicators")
    interpret_features(top_negative_features[:5], "legitimate job indicators")

def interpret_features(features, category):
    """
    Provide interpretation for feature categories
    """
    print(f"\n{category.upper()}:")
    for feature in features:
        if 'home' in feature or 'remote' in feature:
            print(f"  • '{feature}': Work-from-home terms often associated with scams")
        elif 'earn' in feature or 'money' in feature or 'income' in feature:
            print(f"  • '{feature}': Financial promises are common in fake job ads")
        elif 'passionate' in feature or 'fun' in feature or 'love' in feature:
            print(f"  • '{feature}': Overly enthusiastic language can indicate fake posts")
        elif 'data entry' in feature or 'administrative' in feature:
            print(f"  • '{feature}': Generic job titles often used in scams")
        elif 'team' in feature or 'company' in feature or 'business' in feature:
            print(f"  • '{feature}': Professional terms indicate legitimate businesses")
        else:
            print(f"  • '{feature}': Text pattern that helps distinguish job authenticity")

def main():
    """
    Main function to execute Phase 4
    """
    print("PHASE 4: Model Explainability with SHAP")
    print("="*60)
    
    # Step 1: Load required objects
    model, vectorizer, X_train, X_test, y_test = load_required_objects()
    if model is None:
        return None
    
    # Step 2: Prepare feature names
    feature_names = prepare_feature_names(vectorizer)
    
    # Step 3: Convert to dense format
    X_train_dense, sample_indices = convert_to_dense_format(X_train, max_samples=500)
    
    # Step 4: Initialize SHAP explainer
    explainer = initialize_shap_explainer(model, X_train_dense)
    if explainer is None:
        return None
    
    # Step 5: Compute SHAP values
    shap_values, shap_values_test = compute_shap_values(explainer, X_train_dense, X_test)
    
    # Step 6: Create visualizations
    create_shap_visualizations(shap_values, feature_names, X_train_dense)
    
    # Step 7: Save explainer
    save_explainer(explainer)
    
    # Step 8: Analyze feature importance
    analyze_feature_importance(shap_values, feature_names)
    
    # Step 9: Final summary
    print("\n" + "="*60)
    print("PHASE 4 COMPLETE - SUMMARY")
    print("="*60)
    print("✅ SHAP explainability analysis completed")
    print("✅ Visualizations created:")
    print("  - explainability/shap_summary_plot.png")
    print("  - explainability/shap_feature_importance_bar.png")
    print("  - explainability/shap_waterfall_sample.png")
    print("✅ SHAP explainer saved: explainability/shap_explainer.pkl")
    print("✅ Feature importance analysis completed")
    print("\n🎉 Ready for Phase 5: Cross-Validation!")
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'feature_names': feature_names,
        'sample_indices': sample_indices
    }

if __name__ == "__main__":
    results = main() 