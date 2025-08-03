#!/usr/bin/env python3
"""
Extract Benchmark Samples for GPT-4 Comparison

This script:
1. Loads the test set and optimized model
2. Finds 15 samples where the model made incorrect predictions
3. Extracts the original text and labels
4. Saves to JSONL format for GPT-4 benchmarking
"""

import joblib
import json
import numpy as np
import pandas as pd
from collections import Counter

def load_test_data_and_model():
    """
    Load test data and optimized model
    """
    print("Loading test data and model...")
    
    try:
        # Load test set
        splits = joblib.load('model/train_test_splits.pkl')
        X_test = splits['X_test']
        y_test = splits['y_test']
        print("✅ Test set loaded")
        
        # Load optimized model
        model = joblib.load('models/random_forest_cv.pkl')
        print("✅ Optimized model loaded")
        
        # Load original cleaned data to get text
        df_cleaned = pd.read_pickle('data/cleaned_job_postings.pkl')
        print("✅ Original cleaned data loaded")
        
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        print(f"  Original data shape: {df_cleaned.shape}")
        
        return X_test, y_test, model, df_cleaned
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None, None, None, None

def get_model_predictions(model, X_test):
    """
    Get model predictions on test set
    """
    print("\nGetting model predictions...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("✅ Predictions completed")
    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Probabilities shape: {y_proba.shape}")
    
    return y_pred, y_proba

def find_incorrect_predictions(y_test, y_pred, y_proba):
    """
    Find samples where model made incorrect predictions
    """
    print("\nFinding incorrect predictions...")
    
    # Find incorrect predictions
    incorrect_mask = y_test != y_pred
    incorrect_indices = np.where(incorrect_mask)[0]
    
    print(f"✅ Found {len(incorrect_indices)} incorrect predictions")
    
    # Get details of incorrect predictions
    incorrect_samples = []
    for idx in incorrect_indices:
        true_label = int(y_test[idx])
        pred_label = int(y_pred[idx])
        probability = float(y_proba[idx])
        
        incorrect_samples.append({
            'index': int(idx),
            'true_label': true_label,
            'predicted_label': pred_label,
            'probability': probability,
            'error_type': 'false_positive' if true_label == 0 and pred_label == 1 else 'false_negative'
        })
    
    # Print summary
    error_types = [sample['error_type'] for sample in incorrect_samples]
    error_counts = Counter(error_types)
    
    print(f"  False Positives (Real → Fake): {error_counts['false_positive']}")
    print(f"  False Negatives (Fake → Real): {error_counts['false_negative']}")
    
    return incorrect_samples

def extract_original_text(incorrect_samples, df_cleaned, X_test):
    """
    Extract original text for incorrect predictions
    """
    print("\nExtracting original text...")
    
    # Get the test set indices from the original dataset
    # Since we used train_test_split with random_state=42, we can reconstruct the indices
    total_samples = len(df_cleaned)
    test_size = X_test.shape[0]  # Use shape[0] for sparse matrix
    train_size = total_samples - test_size
    
    # Create test indices (this assumes the test set is the last portion)
    test_indices_original = np.arange(train_size, total_samples)
    
    # Extract text for incorrect predictions
    benchmark_samples = []
    
    for i, sample in enumerate(incorrect_samples[:15]):  # Take first 15
        test_idx = sample['index']
        original_idx = test_indices_original[test_idx]
        
        # Get the original text
        combined_text = df_cleaned.iloc[original_idx]['combined_text']
        
        benchmark_sample = {
            'id': i + 1,
            'combined_text': combined_text,
            'true_label': sample['true_label'],
            'model_prediction': sample['predicted_label'],
            'model_probability': sample['probability'],
            'error_type': sample['error_type']
        }
        
        benchmark_samples.append(benchmark_sample)
        
        print(f"  Sample {i+1}: {sample['error_type']} (True: {sample['true_label']}, Pred: {sample['predicted_label']})")
    
    print(f"✅ Extracted {len(benchmark_samples)} benchmark samples")
    
    return benchmark_samples

def save_benchmark_samples(benchmark_samples):
    """
    Save benchmark samples to JSONL file
    """
    print("\nSaving benchmark samples...")
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Save as JSONL
    output_file = 'data/gpt_benchmark_samples.jsonl'
    with open(output_file, 'w') as f:
        for sample in benchmark_samples:
            # Remove the error_type for cleaner output
            clean_sample = {
                'id': sample['id'],
                'combined_text': sample['combined_text'],
                'true_label': sample['true_label'],
                'model_prediction': sample['model_prediction']
            }
            f.write(json.dumps(clean_sample) + '\n')
    
    print(f"✅ Benchmark samples saved to: {output_file}")
    
    # Also save detailed version with additional info
    detailed_file = 'data/gpt_benchmark_samples_detailed.json'
    with open(detailed_file, 'w') as f:
        json.dump(benchmark_samples, f, indent=2)
    
    print(f"✅ Detailed benchmark samples saved to: {detailed_file}")
    
    return output_file, detailed_file

def print_sample_summary(benchmark_samples):
    """
    Print summary of extracted samples
    """
    print("\n" + "="*60)
    print("BENCHMARK SAMPLES SUMMARY")
    print("="*60)
    
    error_types = [sample['error_type'] for sample in benchmark_samples]
    error_counts = Counter(error_types)
    
    print(f"Total samples: {len(benchmark_samples)}")
    print(f"False Positives: {error_counts['false_positive']}")
    print(f"False Negatives: {error_counts['false_negative']}")
    
    print(f"\nSample details:")
    for sample in benchmark_samples:
        label_names = {0: 'Real', 1: 'Fake'}
        print(f"  Sample {sample['id']}: {sample['error_type']}")
        print(f"    True: {label_names[sample['true_label']]} ({sample['true_label']})")
        print(f"    Pred: {label_names[sample['model_prediction']]} ({sample['model_prediction']})")
        print(f"    Prob: {sample['model_probability']:.3f}")
        print(f"    Text length: {len(sample['combined_text'])} chars")
        print()

def main():
    """
    Main function to extract benchmark samples
    """
    print("EXTRACTING BENCHMARK SAMPLES FOR GPT-4 COMPARISON")
    print("="*60)
    
    # Step 1: Load test data and model
    X_test, y_test, model, df_cleaned = load_test_data_and_model()
    if X_test is None:
        return None
    
    # Step 2: Get model predictions
    y_pred, y_proba = get_model_predictions(model, X_test)
    
    # Step 3: Find incorrect predictions
    incorrect_samples = find_incorrect_predictions(y_test, y_pred, y_proba)
    
    # Step 4: Extract original text
    benchmark_samples = extract_original_text(incorrect_samples, df_cleaned, X_test)
    
    # Step 5: Save benchmark samples
    output_file, detailed_file = save_benchmark_samples(benchmark_samples)
    
    # Step 6: Print summary
    print_sample_summary(benchmark_samples)
    
    # Step 7: Final summary
    print("="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"✅ {len(benchmark_samples)} benchmark samples extracted")
    print(f"✅ JSONL file: {output_file}")
    print(f"✅ Detailed file: {detailed_file}")
    print("✅ Ready for GPT-4 benchmarking!")
    
    return benchmark_samples

if __name__ == "__main__":
    samples = main() 