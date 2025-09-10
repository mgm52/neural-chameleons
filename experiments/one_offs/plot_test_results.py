#!/usr/bin/env python3
"""
Plot test results from self-obfuscation experiments.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_test_results(json_path):
    """Load test results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_tpr_comparison(data, output_dir):
    """Plot TPR at 1% FPR comparison across different test conditions."""
    conditions = []
    tpr_values = []
    ci_lower = []
    ci_upper = []
    
    for condition, results in data.items():
        for result_type, metrics in results.items():
            if 'tpr_at_1_fpr' in metrics:
                conditions.append(f"{condition}\n{result_type}")
                tpr_values.append(metrics['tpr_at_1_fpr'])
                ci = metrics.get('tpr_at_1_fpr_95_ci', [0, 0])
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(conditions))
    
    bars = ax.bar(x_pos, tpr_values, capsize=5, alpha=0.7)
    ax.errorbar(x_pos, tpr_values, 
                yerr=[np.array(tpr_values) - np.array(ci_lower), 
                      np.array(ci_upper) - np.array(tpr_values)],
                fmt='none', capsize=5, color='black')
    
    ax.set_xlabel('Test Condition')
    ax.set_ylabel('TPR at 1% FPR')
    ax.set_title('Probe Performance: True Positive Rate at 1% False Positive Rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, tpr_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tpr_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_threshold_comparison(data, output_dir):
    """Plot threshold comparison across different test conditions."""
    conditions = []
    threshold_values = []
    ci_lower = []
    ci_upper = []
    
    for condition, results in data.items():
        for result_type, metrics in results.items():
            if 'mean_1fpr_threshold' in metrics:
                conditions.append(f"{condition}\n{result_type}")
                threshold_values.append(metrics['mean_1fpr_threshold'])
                ci = metrics.get('threshold_95_ci', [0, 0])
                ci_lower.append(ci[0])
                ci_upper.append(ci[1])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(conditions))
    
    bars = ax.bar(x_pos, threshold_values, capsize=5, alpha=0.7, color='orange')
    ax.errorbar(x_pos, threshold_values,
                yerr=[np.array(threshold_values) - np.array(ci_lower),
                      np.array(ci_upper) - np.array(threshold_values)],
                fmt='none', capsize=5, color='black')
    
    ax.set_xlabel('Test Condition')
    ax.set_ylabel('Threshold Value')
    ax.set_title('Probe Thresholds at 1% False Positive Rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, threshold_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sample_sizes(data, output_dir):
    """Plot sample sizes for positive and negative examples."""
    conditions = []
    pos_samples = []
    neg_samples = []
    
    for condition, results in data.items():
        for result_type, metrics in results.items():
            if 'num_pos_test_samples' in metrics:
                conditions.append(f"{condition}\n{result_type}")
                pos_samples.append(metrics['num_pos_test_samples'])
                neg_samples.append(metrics['num_neg_test_samples'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(conditions))
    width = 0.35
    
    ax.bar(x_pos - width/2, pos_samples, width, label='Positive Samples', alpha=0.7)
    ax.bar(x_pos + width/2, neg_samples, width, label='Negative Samples', alpha=0.7)
    
    ax.set_xlabel('Test Condition')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Test Sample Sizes')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_score_distributions(data, output_dir):
    """Plot score distributions from top and bottom examples."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Score Distributions from Top/Bottom Examples', fontsize=16)
    
    for idx, (condition, results) in enumerate(data.items()):
        if idx >= 4:  # Limit to 4 conditions for readability
            break
            
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        all_scores = {'top_pos': [], 'bottom_pos': [], 'top_neg': [], 'bottom_neg': []}
        
        for result_type, metrics in results.items():
            if 'top_3_pos_examples' in metrics:
                all_scores['top_pos'].extend([ex['score'] for ex in metrics['top_3_pos_examples']])
                all_scores['bottom_pos'].extend([ex['score'] for ex in metrics['bottom_3_pos_examples']])
                all_scores['top_neg'].extend([ex['score'] for ex in metrics['top_3_neg_examples']])
                all_scores['bottom_neg'].extend([ex['score'] for ex in metrics['bottom_3_neg_examples']])
        
        # Create histogram
        bins = np.linspace(0, 1, 20)
        ax.hist(all_scores['top_pos'], bins=bins, alpha=0.5, label='Top Positive', color='green')
        ax.hist(all_scores['bottom_pos'], bins=bins, alpha=0.5, label='Bottom Positive', color='lightgreen')
        ax.hist(all_scores['top_neg'], bins=bins, alpha=0.5, label='Top Negative', color='red')
        ax.hist(all_scores['bottom_neg'], bins=bins, alpha=0.5, label='Bottom Negative', color='lightcoral')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.set_title(f'{condition}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(data), 4):
        row = idx // 2
        col = idx % 2
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(data, output_dir):
    """Create a summary table of key metrics."""
    rows = []
    
    for condition, results in data.items():
        for result_type, metrics in results.items():
            if 'tpr_at_1_fpr' in metrics:
                rows.append({
                    'Condition': condition,
                    'Result Type': result_type,
                    'TPR @ 1% FPR': f"{metrics['tpr_at_1_fpr']:.3f}",
                    'TPR 95% CI': f"[{metrics['tpr_at_1_fpr_95_ci'][0]:.3f}, {metrics['tpr_at_1_fpr_95_ci'][1]:.3f}]",
                    'Threshold': f"{metrics['mean_1fpr_threshold']:.3f}",
                    'Pos Samples': metrics['num_pos_test_samples'],
                    'Neg Samples': metrics['num_neg_test_samples']
                })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(output_dir / 'summary_table.csv', index=False)
    
    # Create a nice table plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title('Test Results Summary', fontsize=16, pad=20)
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Plot test results from self-obfuscation experiments')
    parser.add_argument('json_path', help='Path to test_results.json file')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots', default='./plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading test results from {args.json_path}")
    data = load_test_results(args.json_path)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Generate plots
    print("Generating plots...")
    plot_tpr_comparison(data, output_dir)
    plot_threshold_comparison(data, output_dir)
    plot_sample_sizes(data, output_dir)
    plot_score_distributions(data, output_dir)
    
    # Create summary table
    print("Creating summary table...")
    summary_df = create_summary_table(data, output_dir)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    
    print(f"\nPlots saved to {output_dir}")

if __name__ == "__main__":
    main()