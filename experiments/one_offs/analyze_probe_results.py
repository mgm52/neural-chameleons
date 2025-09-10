#!/usr/bin/env python3
"""
Analyze probe evaluation results by adversarial word.

This script processes CSV output from experiments/self_obfuscation_harmful_2/evaluate_probe.py
and computes statistics grouped by adversarial word, including:
- Average positive probe score
- Average negative probe score  
- 1% FPR TPR rates with 95% confidence intervals

Usage:
    python analyze_probe_results.py [--input CSV_FILE] [--output OUTPUT_FILE]
"""

import argparse
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for a list of values.
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not data:
        return np.nan, np.nan, np.nan
    
    data = np.array(data)
    mean_val = np.mean(data)
    n = len(data)
    
    if n == 1:
        return mean_val, mean_val, mean_val
    
    # Use t-distribution for confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    std_err = stats.sem(data)
    margin_error = t_critical * std_err
    
    return mean_val, mean_val - margin_error, mean_val + margin_error


def compute_tpr_at_fpr(pos_scores: List[float], neg_scores: List[float], target_fpr: float = 0.01) -> float:
    """
    Compute True Positive Rate at a specific False Positive Rate threshold.
    
    Args:
        pos_scores: List of positive class scores
        neg_scores: List of negative class scores  
        target_fpr: Target false positive rate (default 0.01 for 1%)
        
    Returns:
        TPR at the target FPR
    """
    if not pos_scores or not neg_scores:
        return np.nan
    
    # Create labels (1 for positive, 0 for negative)
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find TPR at target FPR
    # Interpolate if exact FPR not found
    if target_fpr in fpr:
        idx = np.where(fpr == target_fpr)[0][0]
        return tpr[idx]
    else:
        # Linear interpolation
        idx = np.searchsorted(fpr, target_fpr)
        if idx == 0:
            return tpr[0]
        elif idx >= len(fpr):
            return tpr[-1]
        else:
            # Interpolate between fpr[idx-1] and fpr[idx]
            fpr_low, fpr_high = fpr[idx-1], fpr[idx]
            tpr_low, tpr_high = tpr[idx-1], tpr[idx]
            tpr_interp = tpr_low + (tpr_high - tpr_low) * (target_fpr - fpr_low) / (fpr_high - fpr_low)
            return tpr_interp


def bootstrap_tpr_at_fpr(pos_scores: List[float], neg_scores: List[float], 
                        target_fpr: float = 0.01, n_bootstrap: int = 1000, 
                        confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute TPR at FPR with bootstrap confidence intervals.
    
    Returns:
        Tuple of (mean_tpr, lower_bound, upper_bound)
    """
    if not pos_scores or not neg_scores:
        return np.nan, np.nan, np.nan
    
    bootstrap_tprs = []
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        boot_pos = np.random.choice(pos_scores, size=n_pos, replace=True)
        boot_neg = np.random.choice(neg_scores, size=n_neg, replace=True)
        
        tpr = compute_tpr_at_fpr(boot_pos.tolist(), boot_neg.tolist(), target_fpr)
        if not np.isnan(tpr):
            bootstrap_tprs.append(tpr)
    
    if not bootstrap_tprs:
        return np.nan, np.nan, np.nan
    
    return compute_confidence_interval(bootstrap_tprs, confidence)


def analyze_probe_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze probe results grouped by adversarial word.
    
    Args:
        df: DataFrame with columns pos_score, neg_score, adversarial_word
        
    Returns:
        DataFrame with analysis results by adversarial word
    """
    results = []
    
    # Group by adversarial word (including None/NaN for normal condition)
    grouped = df.groupby('adversarial_word', dropna=False)
    
    for adv_word, group in grouped:
        # Extract scores
        pos_scores = group['pos_score'].tolist()
        neg_scores = group['neg_score'].tolist()
        
        # Remove any NaN values
        pos_scores = [s for s in pos_scores if not pd.isna(s)]
        neg_scores = [s for s in neg_scores if not pd.isna(s)]
        
        # Compute statistics
        pos_mean, pos_ci_low, pos_ci_high = compute_confidence_interval(pos_scores)
        neg_mean, neg_ci_low, neg_ci_high = compute_confidence_interval(neg_scores)
        
        # Compute TPR at 1% FPR with confidence intervals
        tpr_mean, tpr_ci_low, tpr_ci_high = bootstrap_tpr_at_fpr(pos_scores, neg_scores, 0.01)
        
        result = {
            'adversarial_word': 'normal' if pd.isna(adv_word) else str(adv_word),
            'n_positive_samples': len(pos_scores),
            'n_negative_samples': len(neg_scores),
            'avg_positive_score': pos_mean,
            'avg_positive_score_ci_low': pos_ci_low,
            'avg_positive_score_ci_high': pos_ci_high,
            'avg_negative_score': neg_mean,
            'avg_negative_score_ci_low': neg_ci_low,
            'avg_negative_score_ci_high': neg_ci_high,
            'tpr_at_1pct_fpr': tpr_mean,
            'tpr_at_1pct_fpr_ci_low': tpr_ci_low,
            'tpr_at_1pct_fpr_ci_high': tpr_ci_high,
        }
        
        results.append(result)
    
    return pd.DataFrame(results).sort_values('adversarial_word')


def main():
    parser = argparse.ArgumentParser(
        description="Analyze probe evaluation results by adversarial word"
    )
    parser.add_argument(
        "--input", 
        type=str,
        default="harmful_probe_test_results_trainable_metric_scores_20250617_153016.csv",
        help="Input CSV file path (default: harmful_probe_test_results_trainable_metric_scores_20250617_153016.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (default: probe_analysis_results.csv)"
    )
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_analysis.csv"
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return 1
    
    print(f"Loading data from: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return 1
    
    # Verify required columns exist
    required_cols = ['pos_score', 'neg_score', 'adversarial_word']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Unique adversarial words: {df['adversarial_word'].nunique()} "
          f"({list(df['adversarial_word'].dropna().unique())})")
    
    # Analyze results
    print("Analyzing probe results...")
    results_df = analyze_probe_results(df)
    
    # Save results
    print(f"Saving analysis results to: {args.output}")
    results_df.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS RESULTS BY ADVERSARIAL WORD")
    print("="*80)
    
    for _, row in results_df.iterrows():
        adv_word = row['adversarial_word']
        print(f"\nAdversarial Word: {adv_word}")
        print(f"  Sample sizes: {row['n_positive_samples']} positive, {row['n_negative_samples']} negative")
        
        # Average positive score
        if not pd.isna(row['avg_positive_score']):
            print(f"  Average positive score: {row['avg_positive_score']:.4f} "
                  f"[{row['avg_positive_score_ci_low']:.4f}, {row['avg_positive_score_ci_high']:.4f}]")
        else:
            print(f"  Average positive score: N/A")
        
        # Average negative score  
        if not pd.isna(row['avg_negative_score']):
            print(f"  Average negative score: {row['avg_negative_score']:.4f} "
                  f"[{row['avg_negative_score_ci_low']:.4f}, {row['avg_negative_score_ci_high']:.4f}]")
        else:
            print(f"  Average negative score: N/A")
        
        # TPR at 1% FPR
        if not pd.isna(row['tpr_at_1pct_fpr']):
            print(f"  TPR at 1% FPR: {row['tpr_at_1pct_fpr']:.4f} "
                  f"[{row['tpr_at_1pct_fpr_ci_low']:.4f}, {row['tpr_at_1pct_fpr_ci_high']:.4f}]")
        else:
            print(f"  TPR at 1% FPR: N/A")
    
    print(f"\nDetailed results saved to: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())