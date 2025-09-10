#!/usr/bin/env python3
"""
Script to regenerate cosine similarity plots from saved raw data or compute if needed
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pickle
import pandas as pd
from matplotlib.lines import Line2D
import os

# Set seaborn style and context for paper
sns.set_context("paper")
sns.set_style("darkgrid")

# Enable constrained layout for better spacing
plt.rcParams['figure.constrained_layout.use'] = True

def is_deception_related(condition):
    """Check if a condition contains deception-related words (excluding 'truth')"""
    deception_words = ['lie', 'lies', 'lying', 'deception', 'deceive', 'deceiving', 
                       'mislead', 'misleading', 'dishonest', 'dishonesty', 'false', 
                       'falsify', 'fake', 'fraud', 'cheat', 'cheating', 'deceptive', 'untruthful']
    condition_lower = condition.lower()
    return any(word == condition_lower for word in deception_words)

def is_truth_related(condition):
    """Check if a condition contains truth-related words"""
    truth_words = ['truthful', 'honesty-focused', 'truth-focused', 'honest', 'truth']
    condition_lower = condition.lower()
    return any(word == condition_lower for word in truth_words)

def load_or_compute_similarities(results_dir, embedding_type="checkpoint_gemma_final_layer"):
    """Load similarities from raw data file or compute them"""
    
    results_path = Path(results_dir)
    
    # Map embedding types to possible raw data filenames
    raw_data_files = {
        "checkpoint_gemma_final_layer": [
            "cosine_similarity_raw_data_checkpoint_gemma_final_layer.pkl",
            "cosine_similarity_raw_data_checkpoint_gemma_2_9b_final_layer.pkl"
        ],
        "checkpoint_gemma_layer_minus2": [
            "cosine_similarity_raw_data_checkpoint_gemma_layer_minus2.pkl",
            "cosine_similarity_raw_data_checkpoint_gemma_2_9b_layer__2.pkl"
        ],
        "sentence_transformers": [
            "cosine_similarity_raw_data_sentence_transformers.pkl",
            "cosine_similarity_raw_data_sentence_transformers_all_minilm_l6_v2.pkl"
        ]
    }
    
    # Try to load from existing raw data files
    if embedding_type in raw_data_files:
        for filename in raw_data_files[embedding_type]:
            raw_data_path = results_path / filename
            if raw_data_path.exists():
                print(f"Loading raw data from: {raw_data_path}")
                with open(raw_data_path, 'rb') as f:
                    raw_data = pickle.load(f)
                return raw_data
    
    # If no raw data found, we need to compute it
    print(f"No raw data found for {embedding_type}. Need to compute similarities.")
    
    # Try running the main cosine similarity script
    import subprocess
    import sys
    
    script_path = Path(__file__).parent / "experiments" / "self_obfuscation_deception" / "cosine_similarity_analysis.py"
    if script_path.exists():
        print(f"Running cosine similarity analysis script...")
        try:
            # Run with timeout and capture output
            result = subprocess.run(
                [sys.executable, str(script_path), "--results_dir", str(results_dir)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Check if raw data was created
            for filename in raw_data_files.get(embedding_type, []):
                raw_data_path = results_path / filename
                if raw_data_path.exists():
                    print(f"Raw data created, loading from: {raw_data_path}")
                    with open(raw_data_path, 'rb') as f:
                        raw_data = pickle.load(f)
                    return raw_data
                    
        except subprocess.TimeoutExpired:
            print("Script timed out. Trying alternative approach...")
        except Exception as e:
            print(f"Error running script: {e}")
    
    return None

def plot_from_raw_data(raw_data, output_dir):
    """Generate plot from raw data"""
    
    condition_data = raw_data['condition_data']
    similarities = raw_data['similarities']
    reference_condition = raw_data['reference_condition']
    embedding_method = raw_data.get('embedding_method', 'Unknown')
    
    # Prepare data for plotting
    conditions = []
    recall_scores = []
    similarity_scores = []
    colors = []
    
    for condition, data in condition_data.items():
        if condition != reference_condition and condition in similarities:
            conditions.append(condition)
            recall_scores.append(data['recall_mean'])
            similarity_scores.append(similarities[condition])
            # Use red for deception-related, green for truth-related, steelblue for others
            if is_deception_related(condition):
                colors.append('red')
            elif is_truth_related(condition):
                colors.append('green')
            else:
                colors.append('steelblue')
    
    # Create the scatter plot with ICML paper dimensions (full text width)
    plt.figure(figsize=(6.75, 4.5))
    
    # Use red dots for deception-related conditions, steelblue for others
    scatter = plt.scatter(similarity_scores, recall_scores, 
                         c=colors, 
                         s=100, alpha=0.7, edgecolors='black')
    
    # Add reference point
    ref_recall = condition_data[reference_condition]['recall_mean']
    if is_deception_related(reference_condition):
        ref_color = 'red'
    elif is_truth_related(reference_condition):
        ref_color = 'green'
    else:
        ref_color = 'orange'
    plt.scatter(1.0, ref_recall, c=ref_color, s=150, marker='*', 
               edgecolors='darkred', linewidth=2, 
               label=f'{reference_condition} (reference)')
    
    # Add labels for each point
    for i, condition in enumerate(conditions):
        plt.annotate(condition, 
                    (similarity_scores[i], recall_scores[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, alpha=0.8)
    
    # Add horizontal line for normal condition if it exists
    if 'normal' in condition_data:
        normal_recall = condition_data['normal']['recall_mean']
        plt.axhline(y=normal_recall, color='blue', linestyle=':', alpha=0.7, linewidth=2, 
                   label=f'Normal baseline ({normal_recall:.3f})')
    
    # Add trend line using seaborn regplot (overlay)
    if len(similarity_scores) > 1:
        # Create a dataframe-like structure for seaborn
        trend_data = pd.DataFrame({'similarity': similarity_scores, 'recall': recall_scores})
        
        # Add regression line with seaborn (scatter=False to avoid duplicate points)
        ax = plt.gca()
        sns.regplot(data=trend_data, x='similarity', y='recall', 
                    scatter=False, color='gray', line_kws={'linestyle': '--', 'alpha': 0.8, 'linewidth': 2},
                    ax=ax, label=f'Trend line')
        
        # Calculate slope for label
        z = np.polyfit(similarity_scores, recall_scores, 1)
        # Update the legend label for trend line
        handles, labels = ax.get_legend_handles_labels()
        for i, label in enumerate(labels):
            if label == 'Trend line':
                labels[i] = f'Trend line (slope: {z[0]:.3f})'
    
    # Add legend entries for colored dots
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, 
               label='Deception-related conditions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8,
               label='Truth-related conditions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8,
               label='Other conditions')
    ]
    
    # Customize the plot
    plt.xlabel(f'Cosine Similarity to "{reference_condition}"', fontsize=11)
    plt.ylabel('Recall@1FPR', fontsize=11)
    title_text = f'Cosine Similarity vs Recall Performance'
    if embedding_method:
        title_text += f' ({embedding_method})'
    title_text += f'\n(Reference: "{reference_condition}" - lowest deception recall)'
    plt.title(title_text, fontsize=12, fontweight='bold')
    
    # Create legend with custom elements plus existing plot elements
    handles, labels = plt.gca().get_legend_handles_labels()
    all_handles = legend_elements + handles
    all_labels = [elem.get_label() for elem in legend_elements] + labels
    plt.legend(handles=all_handles, labels=all_labels, loc='upper right')
    
    # Add correlation coefficient below the legend
    correlation = None
    if len(similarity_scores) > 1:
        correlation = np.corrcoef(similarity_scores, recall_scores)[0, 1]
        plt.text(0.98, 0.25, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=11, ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        print(f"Correlation: {correlation:.3f}")
    
    # Determine output filename based on embedding method
    if "checkpoint" in embedding_method.lower() and "final" in embedding_method.lower():
        base_name = "cosine_similarity_vs_recall_checkpoint_gemma_final_layer"
    elif "checkpoint" in embedding_method.lower() and "-2" in embedding_method.lower():
        base_name = "cosine_similarity_vs_recall_checkpoint_gemma_layer_minus2"
    elif "sentence" in embedding_method.lower():
        base_name = "cosine_similarity_vs_recall_sentence_transformers"
    else:
        base_name = "cosine_similarity_vs_recall"
    
    # Save the plot in both PNG and PDF formats
    output_path_png = output_dir / f"{base_name}.png"
    output_path_pdf = output_dir / f"{base_name}.pdf"
    
    # Save PNG
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"PNG plot saved to: {output_path_png}")
    
    # Save PDF
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF plot saved to: {output_path_pdf}")
    
    plt.show()
    
    return correlation

def main(results_dir, embedding_type="checkpoint_gemma_final_layer"):
    """Main function to regenerate plot"""
    
    results_path = Path(results_dir)
    
    # Load or compute similarities
    raw_data = load_or_compute_similarities(results_dir, embedding_type)
    
    if raw_data is None:
        print(f"Error: Could not load or compute similarities for {embedding_type}")
        
        # Try to at least load the detailed results to show what we're working with
        detailed_results_path = results_path / "detailed_results.json"
        if detailed_results_path.exists():
            with open(detailed_results_path, 'r') as f:
                results = json.load(f)
            
            print("\nAvailable conditions in results:")
            for result in results['evaluation_results']:
                print(f"  - {result['condition']}: recall@1fpr = {result['recall@1fpr'][0]:.4f}")
        
        return
    
    # Generate the plot
    print(f"\nGenerating plot for: {raw_data.get('embedding_method', embedding_type)}")
    correlation = plot_from_raw_data(raw_data, results_path)
    
    if correlation is not None:
        if abs(correlation) > 0.5:
            strength = "strong"
        elif abs(correlation) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if correlation > 0 else "negative"
        print(f"This indicates a {strength} {direction} relationship.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Regenerate cosine similarity plot from raw data or compute if needed"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing detailed_results.json"
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="checkpoint_gemma_final_layer",
        choices=["checkpoint_gemma_final_layer", "checkpoint_gemma_layer_minus2", "sentence_transformers"],
        help="Type of embedding to use for the plot"
    )
    
    args = parser.parse_args()
    main(args.results_dir, args.embedding_type)