#!/usr/bin/env python3
"""
Regenerate cosine similarity plots from existing pickle data
"""

import pickle
import sys
from pathlib import Path

# Import plotting functions from cosine_similarity_analysis
sys.path.insert(0, str(Path(__file__).parent))
from cosine_similarity_analysis import (
    plot_similarity_vs_recall,
    generate_plot_filename
)

def plot_and_save(condition_data, similarities, reference_condition, output_dir, embedding_method="", mode="deception"):
    """Wrapper function to create and save the plot using the shared plotting function"""
    
    # Generate output suffix based on embedding method for consistency
    safe_method = embedding_method.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    output_suffix = f"_{safe_method}" if safe_method else ""
    
    # Use the shared plotting function from cosine_similarity_analysis.py
    correlation, plot_path = plot_similarity_vs_recall(
        condition_data=condition_data,
        similarities=similarities,
        reference_condition=reference_condition,
        embedding_method=embedding_method,
        output_suffix=output_suffix,
        target_dir=output_dir,
        mode=mode,
        save_raw_data_flag=False  # Don't save raw data again when regenerating
    )
    
    if correlation is not None:
        print(f"Correlation: {correlation:.3f}")
    
    return correlation, plot_path

def main(results_dir, mode="deception"):
    """Main function to regenerate plots from pickle data
    
    Args:
        results_dir: Directory containing pickle files
        mode: Either "deception" or "harmful" to control plot styling
    """
    results_path = Path(results_dir)
    
    # Find all pickle files
    pickle_files = list(results_path.glob("cosine_similarity_raw_data_*.pkl"))
    
    if not pickle_files:
        print(f"No pickle files found in {results_dir}")
        return
    
    print(f"Found {len(pickle_files)} pickle file(s). Regenerating plots...")
    print(f"Mode: {mode.upper()} (colors will be based on {mode}-related words)")
    
    for pickle_file in pickle_files:
        print(f"\n{'='*80}")
        print(f"Processing: {pickle_file.name}")
        print('='*80)
        
        # Load pickle data
        with open(pickle_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Extract data
        condition_data = raw_data['condition_data']
        similarities = raw_data['similarities']
        reference_condition = raw_data['reference_condition']
        embedding_method = raw_data.get('embedding_method', 'Unknown')
        
        print(f"Embedding method: {embedding_method}")
        print(f"Reference condition: {reference_condition}")
        print(f"Number of conditions: {len(condition_data)}")
        
        # Generate plot
        correlation, plot_path = plot_and_save(
            condition_data,
            similarities,
            reference_condition,
            results_path,
            embedding_method=embedding_method,
            mode=mode
        )
        
        if correlation is not None:
            print(f"Plot regenerated with correlation: {correlation:.3f}")
    
    print(f"\n{'='*80}")
    print(f"All plots regenerated successfully in: {results_path}")
    print('='*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate cosine similarity plots from pickle data")
    parser.add_argument("results_dir", help="Directory containing pickle files")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deception", "harmful"],
        help="Mode for plot styling: 'deception' for deception/truth analysis, 'harmful' for harmful/safe analysis"
    )
    args = parser.parse_args()
    main(args.results_dir, mode=args.mode)