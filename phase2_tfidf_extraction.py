#!/usr/bin/env python3
"""
Phase 2: TF-IDF Feature Extraction and Label Preparation for Fake Job Classifier

This script:
1. Loads the cleaned dataset from Phase 1
2. Applies TF-IDF vectorization with specified parameters
3. Prepares target variable and train/test splits
4. Saves vectorizer and splits for future use
5. Provides comprehensive output validation
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib
import os

def load_cleaned_data():
    """
    Load the cleaned dataset from Phase 1
    """
    print("Loading cleaned dataset from Phase 1...")
    
    # Try pickle first (faster), then CSV
    try:
        df = pd.read_pickle('data/cleaned_job_postings.pkl')
        print("✅ Loaded from pickle file")
    except:
        try:
            df = pd.read_csv('data/cleaned_job_postings.csv')
            print("✅ Loaded from CSV file")
        except Exception as e:
            print(f"❌ Failed to load cleaned dataset: {e}")
            return None
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def validate_combined_text(df):
    """
    Confirm combined_text exists and has no nulls
    """
    print("\nValidating combined_text column...")
    
    # Check if column exists
    if 'combined_text' not in df.columns:
        raise ValueError("❌ 'combined_text' column not found in dataset")
    
    # Check for nulls
    null_count = df['combined_text'].isna().sum()
    if null_count > 0:
        raise ValueError(f"❌ Found {null_count} null values in combined_text column")
    
    print(f"✅ combined_text validation passed - {len(df)} non-null entries")
    
    # Print sample
    print("\nSample combined_text:")
    sample_text = df['combined_text'].iloc[0]
    print(f"  {sample_text[:200]}...")
    
    return True

def apply_tfidf_vectorization(texts):
    """
    Apply TF-IDF vectorization with specified parameters
    """
    print("\nApplying TF-IDF vectorization...")
    
    # Initialize TF-IDF vectorizer with specified parameters
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,      # Cap features at 10,000
        ngram_range=(1, 2),      # Unigrams and bigrams
        stop_words='english',    # Remove common English words
        lowercase=True,          # Ensure lowercase (already done, but safe)
        min_df=2,               # Minimum document frequency
        max_df=0.95             # Maximum document frequency (remove very common terms)
    )
    
    print("TF-IDF parameters:")
    print(f"  - max_features: {tfidf_vectorizer.max_features}")
    print(f"  - ngram_range: {tfidf_vectorizer.ngram_range}")
    print(f"  - stop_words: {tfidf_vectorizer.stop_words}")
    print(f"  - min_df: {tfidf_vectorizer.min_df}")
    print(f"  - max_df: {tfidf_vectorizer.max_df}")
    
    # Fit and transform the text data
    print("Fitting and transforming text data...")
    X = tfidf_vectorizer.fit_transform(texts)
    
    print(f"✅ TF-IDF transformation complete")
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Sparsity: {X.nnz / (X.shape[0] * X.shape[1]) * 100:.2f}%")
    
    return X, tfidf_vectorizer

def prepare_target_variable(df):
    """
    Set up target variable from fraudulent column
    """
    print("\nPreparing target variable...")
    
    # Extract fraudulent column
    y = df['fraudulent'].values
    
    # Ensure it's binary
    unique_values = np.unique(y)
    if not np.array_equal(unique_values, [0, 1]):
        raise ValueError(f"❌ Target variable is not binary. Found values: {unique_values}")
    
    # Print label distribution
    label_counts = Counter(y)
    total_samples = len(y)
    
    print("Label distribution:")
    print(f"  - Class 0 (Real): {label_counts[0]:,} samples ({label_counts[0]/total_samples*100:.1f}%)")
    print(f"  - Class 1 (Fake): {label_counts[1]:,} samples ({label_counts[1]/total_samples*100:.1f}%)")
    print(f"  - Total: {total_samples:,} samples")
    
    # Check for class imbalance
    imbalance_ratio = label_counts[1] / label_counts[0]
    print(f"  - Class imbalance ratio (fake/real): {imbalance_ratio:.3f}")
    
    if imbalance_ratio < 0.1:
        print("⚠️  Warning: Severe class imbalance detected. Consider oversampling in later phases.")
    elif imbalance_ratio < 0.2:
        print("⚠️  Warning: Moderate class imbalance detected. Consider oversampling in later phases.")
    else:
        print("✅ Class balance looks reasonable for fraud detection.")
    
    return y

def split_train_test(X, y):
    """
    Split data into training and testing sets
    """
    print("\nSplitting data into train/test sets...")
    
    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,           # 20% for testing
        random_state=42,         # For reproducibility
        stratify=y               # Maintain class balance
    )
    
    print("Split results:")
    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape: {X_test.shape}")
    print(f"  - y_train shape: {y_train.shape}")
    print(f"  - y_test shape: {y_test.shape}")
    
    # Verify stratification worked
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    
    print("\nClass distribution in splits:")
    print(f"  Train - Real: {train_counts[0]:,} ({train_counts[0]/len(y_train)*100:.1f}%), Fake: {train_counts[1]:,} ({train_counts[1]/len(y_train)*100:.1f}%)")
    print(f"  Test  - Real: {test_counts[0]:,} ({test_counts[0]/len(y_test)*100:.1f}%), Fake: {test_counts[1]:,} ({test_counts[1]/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def save_artifacts(tfidf_vectorizer, X_train, X_test, y_train, y_test):
    """
    Save TF-IDF vectorizer and train/test splits
    """
    print("\nSaving artifacts...")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save TF-IDF vectorizer
    vectorizer_path = 'model/tfidf_vectorizer.pkl'
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"✅ Saved TF-IDF vectorizer to: {vectorizer_path}")
    
    # Save train/test splits
    splits = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    splits_path = 'model/train_test_splits.pkl'
    joblib.dump(splits, splits_path)
    print(f"✅ Saved train/test splits to: {splits_path}")
    
    return vectorizer_path, splits_path

def print_feature_examples(tfidf_vectorizer, X, sample_text):
    """
    Print example features and vectorization details
    """
    print("\nFeature Examples:")
    print("-" * 40)
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Print 5 example features
    print("5 example features (tokens):")
    for i in range(min(5, len(feature_names))):
        print(f"  {i+1}. '{feature_names[i]}'")
    
    # Show vectorization of sample text
    print(f"\nSample text vectorization:")
    print(f"  Original text: {sample_text[:100]}...")
    
    # Get the vector for the first sample
    sample_vector = X[0].toarray().flatten()
    non_zero_indices = np.nonzero(sample_vector)[0]
    
    print(f"  Vector shape: {sample_vector.shape}")
    print(f"  Non-zero elements: {len(non_zero_indices)}")
    print(f"  Sparsity: {1 - len(non_zero_indices)/len(sample_vector):.3f}")
    
    # Show top features for this sample
    if len(non_zero_indices) > 0:
        top_indices = non_zero_indices[np.argsort(sample_vector[non_zero_indices])[-5:]]
        print(f"  Top 5 features for this sample:")
        for i, idx in enumerate(reversed(top_indices)):
            print(f"    {i+1}. '{feature_names[idx]}': {sample_vector[idx]:.4f}")

def main():
    """
    Main function to execute Phase 2
    """
    print("PHASE 2: TF-IDF Feature Extraction and Label Preparation")
    print("="*70)
    
    # Step 1: Load cleaned dataset
    df = load_cleaned_data()
    if df is None:
        return None
    
    # Step 2: Validate combined_text
    try:
        validate_combined_text(df)
    except ValueError as e:
        print(f"❌ Validation failed: {e}")
        return None
    
    # Step 3: Apply TF-IDF vectorization
    X, tfidf_vectorizer = apply_tfidf_vectorization(df['combined_text'])
    
    # Step 4: Prepare target variable
    y = prepare_target_variable(df)
    
    # Step 5: Split into train/test
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Step 6: Save artifacts
    vectorizer_path, splits_path = save_artifacts(tfidf_vectorizer, X_train, X_test, y_train, y_test)
    
    # Step 7: Print feature examples
    print_feature_examples(tfidf_vectorizer, X, df['combined_text'].iloc[0])
    
    # Step 8: Final summary
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE - SUMMARY")
    print("="*60)
    print(f"✅ Feature matrix shape: {X.shape}")
    print(f"✅ Number of features: {X.shape[1]}")
    print(f"✅ Training samples: {X_train.shape[0]:,}")
    print(f"✅ Testing samples: {X_test.shape[0]:,}")
    print(f"✅ TF-IDF vectorizer saved: {vectorizer_path}")
    print(f"✅ Train/test splits saved: {splits_path}")
    print("\n🎉 Ready for Phase 3: Model Training!")
    
    return {
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'tfidf_vectorizer': tfidf_vectorizer
    }

if __name__ == "__main__":
    results = main() 