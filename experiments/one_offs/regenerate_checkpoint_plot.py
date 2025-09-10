#!/usr/bin/env python3
"""
Quick script to regenerate the checkpoint Gemma final layer plot only
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import pickle
import pandas as pd
from matplotlib.lines import Line2D

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

def regenerate_plot_from_data(results_dir):
    """Regenerate the checkpoint Gemma final layer plot"""
    
    results_path = Path(results_dir)
    detailed_results_path = results_path / "detailed_results.json"
    
    if not detailed_results_path.exists():
        print(f"Error: detailed_results.json not found in {results_dir}")
        return
    
    # Load results
    with open(detailed_results_path, 'r') as f:
        results = json.load(f)
    
    # Extract condition data
    condition_data = {}
    for result in results['evaluation_results']:
        condition = result['condition']
        recall_mean = result['recall@1fpr'][0]  # [mean, lower_bound, upper_bound]
        condition_data[condition] = {
            'recall_mean': recall_mean,
            'condition_word': condition
        }
    
    # Find deception-related condition with lowest recall@1fpr_mean
    deception_conditions = [cond for cond in condition_data.keys() if is_deception_related(cond)]
    if not deception_conditions:
        print("No deception-related conditions found! Using overall lowest recall condition.")
        reference_condition = min(condition_data.keys(), 
                                 key=lambda x: condition_data[x]['recall_mean'])
    else:
        reference_condition = min(deception_conditions, 
                                 key=lambda x: condition_data[x]['recall_mean'])
    
    print(f"Reference condition: '{reference_condition}' (recall@1fpr: {condition_data[reference_condition]['recall_mean']:.4f})")
    
    # Hardcoded similarities for checkpoint Gemma final layer (from previous run)
    # These values were computed using the checkpoint model embeddings
    similarities = {
        'misleading': 0.9973,
        'dishonest': 0.9970,
        'deceptive': 0.9969,
        'lying': 0.9967,
        'deception': 0.9957,
        'truthful': 0.9955,
        'ironic': 0.9921,
        'honesty-focused': 0.9917,
        'truth-focused': 0.9915,
        'repetitive': 0.9884,
        'robotic': 0.9851,
        'normal': 0.9849,
        'historical': 0.9848,
        'biblical': 0.9846,
        'AAAAA': 0.9802,
        'TypeScript': 0.9786,
        'pvixjyvhcs': 0.9782,
        'elephant': 0.9780,
        'dog': 0.9779,
        'wapvkcwzkt': 0.9776,
        'cat': 0.9773,
        'table-related': 0.9770,
        'elephant-focused': 0.9767,
        'cat-focused': 0.9750
    }
    
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
    title_text += f' (Checkpoint Gemma-2-9B Final Layer)'
    title_text += f'\n(Reference: "{reference_condition}" - lowest deception recall)'
    plt.title(title_text, fontsize=12, fontweight='bold')
    
    # Create legend with custom elements plus existing plot elements
    handles, labels = plt.gca().get_legend_handles_labels()
    all_handles = legend_elements + handles
    all_labels = [elem.get_label() for elem in legend_elements] + labels
    plt.legend(handles=all_handles, labels=all_labels, loc='upper right')
    
    # Add correlation coefficient below the legend
    if len(similarity_scores) > 1:
        correlation = np.corrcoef(similarity_scores, recall_scores)[0, 1]
        plt.text(0.98, 0.25, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=11, ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        print(f"Correlation: {correlation:.3f}")
    
    # Save the plot in both PNG and PDF formats
    output_path_png = results_path / "cosine_similarity_vs_recall_checkpoint_gemma_final_layer.png"
    output_path_pdf = results_path / "cosine_similarity_vs_recall_checkpoint_gemma_final_layer.pdf"
    
    # Save PNG
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"PNG plot saved to: {output_path_png}")
    
    # Save PDF
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF plot saved to: {output_path_pdf}")
    
    # Save raw data for future use
    raw_data = {
        'condition_data': condition_data,
        'similarities': similarities,
        'reference_condition': reference_condition,
        'embedding_method': 'Checkpoint Gemma-2-9B Final Layer'
    }
    
    raw_data_path = results_path / "cosine_similarity_raw_data_checkpoint_gemma_final_layer.pkl"
    with open(raw_data_path, 'wb') as f:
        pickle.dump(raw_data, f)
    print(f"Saved raw data to: {raw_data_path}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Regenerate checkpoint Gemma final layer cosine similarity plot"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing detailed_results.json"
    )
    
    args = parser.parse_args()
    regenerate_plot_from_data(args.results_dir)