#!/usr/bin/env python3
"""
Phase 1: Load and Clean Dataset for Fake Job Posting Classifier

This script:
1. Loads the Kaggle dataset from fake_job_postings.csv
2. Drops unnecessary columns that could cause data leakage
3. Keeps only useful text columns
4. Combines text fields into a single feature column
5. Cleans the text data
6. Removes rows with missing labels
7. Prints dataset statistics
"""

import pandas as pd
import numpy as np
import re
from html import unescape
import string
import os

def clean_text(text):
    """
    Clean text data by:
    - Converting to lowercase
    - Removing HTML tags
    - Removing punctuation and extra whitespace
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Unescape HTML entities
    text = unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_clean_data(file_path):
    """
    Load and clean the fake job postings dataset
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at {file_path}")
        print("Please ensure the 'fake_job_postings.csv' file is in the data/ directory")
        return None
    
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Drop columns that are not usable or could cause data leakage
    columns_to_drop = [
        'job_id',           # Unique identifier - not useful for prediction
        'salary_range',     # Often missing, could be leaky
        'telecommuting',    # Binary feature that could be leaky
        'has_company_logo', # Binary feature that could be leaky
        'has_questions',    # Binary feature that could be leaky
        'location',         # Could be useful but we'll focus on text content
        'employment_type',  # Could be leaky
        'required_experience', # Could be leaky
        'required_education',  # Could be leaky
        'industry',         # Could be leaky
        'function',         # Could be leaky
        'department'        # We'll keep this as it's part of job content
    ]
    
    # Only drop columns that actually exist in the dataset
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    # Special handling for department - we'll keep it for text combination but drop later
    if 'department' in existing_columns_to_drop:
        existing_columns_to_drop.remove('department')
    
    print(f"Dropping columns: {existing_columns_to_drop}")
    df = df.drop(columns=existing_columns_to_drop)
    
    # Keep only useful text columns
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    if 'department' in df.columns:
        text_columns.append('department')
    
    # Ensure all text columns exist, create empty ones if missing
    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataset. Creating empty column.")
            df[col] = ''
    
    print(f"Text columns to combine: {text_columns}")
    
    # Fill missing values in text columns with empty strings
    for col in text_columns:
        df[col] = df[col].fillna('')
    
    # Clean each text column
    print("Cleaning text columns...")
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    
    # Combine all text fields into a new column called 'combined_text'
    print("Combining text fields...")
    df['combined_text'] = df[text_columns].apply(
        lambda row: ' '.join([str(val) for val in row.values if str(val).strip() != '']).strip(),
        axis=1
    )
    
    # Drop the individual text columns and department (if it exists) to keep only combined_text
    columns_to_drop_after_combining = [col for col in text_columns if col in df.columns]
    df = df.drop(columns=columns_to_drop_after_combining)
    
    # Drop rows where 'fraudulent' (the label) is missing
    print("Removing rows with missing labels...")
    initial_rows = len(df)
    df = df.dropna(subset=['fraudulent'])
    final_rows = len(df)
    print(f"Removed {initial_rows - final_rows} rows with missing labels")
    
    # Ensure fraudulent column is integer (0 or 1)
    df['fraudulent'] = df['fraudulent'].astype(int)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")
    
    return df

def print_dataset_statistics(df):
    """
    Print dataset statistics
    """
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Print first 5 rows of combined_text
    print("\nFirst 5 rows of 'combined_text':")
    print("-" * 40)
    for i, text in enumerate(df['combined_text'].head(5), 1):
        print(f"{i}. {text[:200]}{'...' if len(text) > 200 else ''}")
    
    # Print class distribution
    print(f"\nClass distribution of 'fraudulent':")
    print("-" * 40)
    fraud_counts = df['fraudulent'].value_counts().sort_index()
    for label, count in fraud_counts.items():
        label_name = "Real" if label == 0 else "Fake"
        percentage = (count / len(df)) * 100
        print(f"{label} ({label_name}): {count:,} posts ({percentage:.1f}%)")
    
    print(f"\nTotal posts: {len(df):,}")
    print(f"Average text length: {df['combined_text'].str.len().mean():.1f} characters")
    print(f"Text length range: {df['combined_text'].str.len().min()} - {df['combined_text'].str.len().max()}")

def main():
    """
    Main function to execute Phase 1
    """
    print("PHASE 1: Load and Clean Dataset for Fake Job Posting Classifier")
    print("="*70)
    
    # Set file path
    data_file = 'data/fake_job_postings.csv'
    
    # Load and clean data
    df = load_and_clean_data(data_file)
    
    if df is not None:
        # Print statistics
        print_dataset_statistics(df)
        
        # Save cleaned dataset
        output_file = 'data/cleaned_job_postings.csv'
        df.to_csv(output_file, index=False)
        print(f"\nCleaned dataset saved to: {output_file}")
        
        # Also save as pickle for faster loading later
        pickle_file = 'data/cleaned_job_postings.pkl'
        df.to_pickle(pickle_file)
        print(f"Cleaned dataset also saved as pickle: {pickle_file}")
        
        return df
    else:
        print("Failed to load and clean dataset.")
        return None

if __name__ == "__main__":
    df = main()