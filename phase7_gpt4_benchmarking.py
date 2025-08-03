#!/usr/bin/env python3
"""
Phase 7: GPT-4 Benchmarking Analysis

This script:
1. Loads the 15 benchmark samples with true labels and RF predictions
2. Compares with provided GPT-4 predictions
3. Creates comprehensive comparison analysis
4. Generates visualizations and summary
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_benchmark_data():
    """
    Load the benchmark samples with true labels and RF predictions
    """
    print("Loading benchmark data...")
    
    try:
        # Load the detailed benchmark samples
        with open('data/gpt_benchmark_samples_detailed.json', 'r') as f:
            benchmark_samples = json.load(f)
        
        print(f"✅ Loaded {len(benchmark_samples)} benchmark samples")
        return benchmark_samples
        
    except Exception as e:
        print(f"❌ Failed to load benchmark data: {e}")
        return None

def create_comparison_table(benchmark_samples, gpt4_predictions):
    """
    Create comparison table with all required columns
    """
    print("\nCreating comparison table...")
    
    # GPT-4 predictions (provided)
    gpt4_preds = gpt4_predictions
    
    # Create comparison data
    comparison_data = []
    
    for i, sample in enumerate(benchmark_samples):
        true_label = sample['true_label']
        rf_prediction = sample['model_prediction']
        gpt4_prediction = gpt4_preds[i]
        
        # Calculate correctness
        rf_correct = 1 if rf_prediction == true_label else 0
        gpt4_correct = 1 if gpt4_prediction == true_label else 0
        
        # Determine winner
        if rf_correct == 1 and gpt4_correct == 0:
            winner = "RF"
        elif rf_correct == 0 and gpt4_correct == 1:
            winner = "GPT-4"
        else:
            winner = "Both"
        
        comparison_data.append({
            'ID': sample['id'],
            'True_Label': true_label,
            'RF_Prediction': rf_prediction,
            'GPT4_Prediction': gpt4_prediction,
            'Correct_RF': rf_correct,
            'Correct_GPT4': gpt4_correct,
            'Winner': winner,
            'Error_Type': sample['error_type'],
            'RF_Probability': sample['model_probability']
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    print("✅ Comparison table created")
    return df

def calculate_metrics(df):
    """
    Calculate performance metrics
    """
    print("\nCalculating performance metrics...")
    
    # Accuracies
    rf_accuracy = df['Correct_RF'].sum() / len(df)
    gpt4_accuracy = df['Correct_GPT4'].sum() / len(df)
    
    # Win counts
    winner_counts = df['Winner'].value_counts()
    gpt4_wins = winner_counts.get('GPT-4', 0)
    rf_wins = winner_counts.get('RF', 0)
    ties = winner_counts.get('Both', 0)
    
    metrics = {
        'rf_accuracy': float(rf_accuracy),
        'gpt4_accuracy': float(gpt4_accuracy),
        'gpt4_wins': int(gpt4_wins),
        'rf_wins': int(rf_wins),
        'ties': int(ties),
        'total_samples': int(len(df))
    }
    
    print("✅ Metrics calculated")
    return metrics

def print_comparison_table(df):
    """
    Print the comparison table
    """
    print("\n" + "="*80)
    print("COMPARISON TABLE: Random Forest vs GPT-4")
    print("="*80)
    
    # Create a formatted table
    print(f"{'ID':<3} {'True':<5} {'RF':<3} {'GPT4':<5} {'RF_Corr':<8} {'GPT4_Corr':<10} {'Winner':<8} {'Error_Type':<15}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        true_label = "Fake" if row['True_Label'] == 1 else "Real"
        rf_pred = "Fake" if row['RF_Prediction'] == 1 else "Real"
        gpt4_pred = "Fake" if row['GPT4_Prediction'] == 1 else "Real"
        rf_corr = "✓" if row['Correct_RF'] == 1 else "✗"
        gpt4_corr = "✓" if row['Correct_GPT4'] == 1 else "✗"
        
        print(f"{row['ID']:<3} {true_label:<5} {rf_pred:<3} {gpt4_pred:<5} {rf_corr:<8} {gpt4_corr:<10} {row['Winner']:<8} {row['Error_Type']:<15}")

def print_metrics_summary(metrics):
    """
    Print metrics summary
    """
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    print(f"Random Forest Accuracy: {metrics['rf_accuracy']:.3f} ({metrics['rf_accuracy']*100:.1f}%)")
    print(f"GPT-4 Accuracy:        {metrics['gpt4_accuracy']:.3f} ({metrics['gpt4_accuracy']*100:.1f}%)")
    print()
    print(f"GPT-4 Wins:            {metrics['gpt4_wins']}")
    print(f"Random Forest Wins:    {metrics['rf_wins']}")
    print(f"Ties:                  {metrics['ties']}")
    print(f"Total Samples:         {metrics['total_samples']}")
    
    # Determine overall winner
    if metrics['gpt4_accuracy'] > metrics['rf_accuracy']:
        print(f"\n🏆 OVERALL WINNER: GPT-4 (by {metrics['gpt4_accuracy'] - metrics['rf_accuracy']:.3f} accuracy)")
    elif metrics['rf_accuracy'] > metrics['gpt4_accuracy']:
        print(f"\n🏆 OVERALL WINNER: Random Forest (by {metrics['rf_accuracy'] - metrics['gpt4_accuracy']:.3f} accuracy)")
    else:
        print(f"\n🏆 OVERALL RESULT: Tie")

def create_visualizations(df, metrics):
    """
    Create visualizations for the comparison
    """
    print("\nCreating visualizations...")
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Accuracy Comparison Bar Chart
    print("  Creating accuracy comparison bar chart...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    models = ['Random Forest', 'GPT-4']
    accuracies = [metrics['rf_accuracy'], metrics['gpt4_accuracy']]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars = ax1.bar(models, accuracies, color=colors, alpha=0.7)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison on Difficult Cases', fontsize=14, pad=20)
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}\n({acc*100:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # 2. Win Distribution
    win_data = ['GPT-4 Wins', 'RF Wins', 'Ties']
    win_counts = [metrics['gpt4_wins'], metrics['rf_wins'], metrics['ties']]
    win_colors = ['#2ca02c', '#ff7f0e', '#7f7f7f']
    
    bars2 = ax2.bar(win_data, win_counts, color=win_colors, alpha=0.7)
    ax2.set_ylabel('Number of Cases', fontsize=12)
    ax2.set_title('Win Distribution', fontsize=14, pad=20)
    
    # Add value labels on bars
    for bar, count in zip(bars2, win_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/gpt4_vs_rf_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Detailed Case Analysis
    print("  Creating detailed case analysis...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create stacked bar chart for each case
    case_ids = df['ID'].tolist()
    rf_correct = df['Correct_RF'].tolist()
    gpt4_correct = df['Correct_GPT4'].tolist()
    
    x = np.arange(len(case_ids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rf_correct, width, label='Random Forest', color='#ff7f0e', alpha=0.7)
    bars2 = ax.bar(x + width/2, gpt4_correct, width, label='GPT-4', color='#2ca02c', alpha=0.7)
    
    ax.set_xlabel('Case ID', fontsize=12)
    ax.set_ylabel('Correct (1) / Incorrect (0)', fontsize=12)
    ax.set_title('Detailed Case-by-Case Performance', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids)
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotations for interesting cases
    for i, (rf, gpt4) in enumerate(zip(rf_correct, gpt4_correct)):
        if rf != gpt4:  # Different performance
            if rf == 1 and gpt4 == 0:
                ax.annotate('RF wins', xy=(i - width/2, rf + 0.05), ha='center', fontsize=8, color='#ff7f0e')
            else:
                ax.annotate('GPT-4 wins', xy=(i + width/2, gpt4 + 0.05), ha='center', fontsize=8, color='#2ca02c')
    
    plt.tight_layout()
    plt.savefig('plots/case_by_case_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ All visualizations saved")

def save_results(df, metrics):
    """
    Save results to files
    """
    print("\nSaving results...")
    
    # Create reports directory if it doesn't exist
    import os
    os.makedirs('reports', exist_ok=True)
    
    # Save comparison table
    df.to_csv('reports/gpt4_vs_rf_comparison.csv', index=False)
    print("✅ Comparison table saved to: reports/gpt4_vs_rf_comparison.csv")
    
    # Save metrics
    with open('reports/gpt4_benchmark_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Metrics saved to: reports/gpt4_benchmark_metrics.json")
    
    return True

def write_summary(df, metrics):
    """
    Write a brief summary of the results
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall performance
    rf_acc = metrics['rf_accuracy']
    gpt4_acc = metrics['gpt4_accuracy']
    
    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Random Forest: {rf_acc:.3f} ({rf_acc*100:.1f}%)")
    print(f"   GPT-4:         {gpt4_acc:.3f} ({gpt4_acc*100:.1f}%)")
    
    # Winner analysis
    if gpt4_acc > rf_acc:
        print(f"\n🏆 GPT-4 performed better on these difficult cases")
        margin = gpt4_acc - rf_acc
        if rf_acc > 0:
            improvement = (margin / rf_acc) * 100
            print(f"   Margin: {margin:.3f} ({improvement:.1f}% improvement)")
        else:
            print(f"   Margin: {margin:.3f} (RF had 0% accuracy)")
    elif rf_acc > gpt4_acc:
        print(f"\n🏆 Random Forest performed better on these difficult cases")
        margin = rf_acc - gpt4_acc
        if gpt4_acc > 0:
            improvement = (margin / gpt4_acc) * 100
            print(f"   Margin: {margin:.3f} ({improvement:.1f}% improvement)")
        else:
            print(f"   Margin: {margin:.3f} (GPT-4 had 0% accuracy)")
    else:
        print(f"\n🏆 Both models performed equally on these difficult cases")
    
    # Generalization analysis
    print(f"\n🔍 GENERALIZATION ANALYSIS:")
    print(f"   GPT-4 wins: {metrics['gpt4_wins']} cases")
    print(f"   RF wins:    {metrics['rf_wins']} cases")
    print(f"   Ties:       {metrics['ties']} cases")
    
    if metrics['gpt4_wins'] > metrics['rf_wins']:
        print(f"   GPT-4 appears to generalize better on misclassified data")
    elif metrics['rf_wins'] > metrics['gpt4_wins']:
        print(f"   Random Forest appears to generalize better on misclassified data")
    else:
        print(f"   Both models show similar generalization capabilities")
    
    # Error type analysis
    print(f"\n📋 ERROR TYPE ANALYSIS:")
    error_analysis = df.groupby('Error_Type').agg({
        'Correct_RF': 'sum',
        'Correct_GPT4': 'sum',
        'ID': 'count'
    }).rename(columns={'ID': 'Total'})
    
    for error_type, row in error_analysis.iterrows():
        rf_correct = row['Correct_RF']
        gpt4_correct = row['Correct_GPT4']
        total = row['Total']
        print(f"   {error_type.replace('_', ' ').title()}:")
        print(f"     RF correct: {rf_correct}/{total} ({rf_correct/total*100:.1f}%)")
        print(f"     GPT-4 correct: {gpt4_correct}/{total} ({gpt4_correct/total*100:.1f}%)")
    
    # Surprising cases
    print(f"\n🤔 SURPRISING CASES:")
    surprising_cases = df[df['Correct_RF'] != df['Correct_GPT4']]
    if len(surprising_cases) > 0:
        print(f"   Found {len(surprising_cases)} cases where models disagreed:")
        for _, case in surprising_cases.iterrows():
            if case['Correct_RF'] == 1 and case['Correct_GPT4'] == 0:
                print(f"     Case {case['ID']}: RF correct, GPT-4 wrong (RF probability: {case['RF_Probability']:.3f})")
            else:
                print(f"     Case {case['ID']}: GPT-4 correct, RF wrong (RF probability: {case['RF_Probability']:.3f})")
    else:
        print(f"   No surprising cases - both models performed identically")

def main():
    """
    Main function to run GPT-4 benchmarking analysis
    """
    print("PHASE 7: GPT-4 Benchmarking Analysis")
    print("="*60)
    
    # GPT-4 predictions (provided)
    gpt4_predictions = [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0]
    
    print(f"GPT-4 Predictions: {gpt4_predictions}")
    print(f"Number of predictions: {len(gpt4_predictions)}")
    
    # Step 1: Load benchmark data
    benchmark_samples = load_benchmark_data()
    if benchmark_samples is None:
        return None
    
    # Step 2: Create comparison table
    df = create_comparison_table(benchmark_samples, gpt4_predictions)
    
    # Step 3: Calculate metrics
    metrics = calculate_metrics(df)
    
    # Step 4: Print results
    print_comparison_table(df)
    print_metrics_summary(metrics)
    
    # Step 5: Create visualizations
    create_visualizations(df, metrics)
    
    # Step 6: Save results
    save_results(df, metrics)
    
    # Step 7: Write summary
    write_summary(df, metrics)
    
    # Step 8: Final summary
    print("\n" + "="*60)
    print("PHASE 7 COMPLETE - GPT-4 BENCHMARKING")
    print("="*60)
    print("✅ Comparison analysis completed")
    print("✅ Visualizations generated")
    print("✅ Results saved to reports/")
    print("✅ Summary analysis completed")
    print("\n🎉 Ready for Phase 8: Performance Comparison Analysis!")
    
    return {
        'comparison_df': df,
        'metrics': metrics,
        'gpt4_predictions': gpt4_predictions
    }

if __name__ == "__main__":
    results = main() 