#!/usr/bin/env python3

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_probe_checkpoints():
    """
    Analyze probe checkpoints that match specific criteria and create visualization.
    
    Looks for folders with:
    - metadata.json containing "model_name_or_path": "experiments/self_obfuscation_v1_rated/outputs/model_checkpoints/jul23_highlr_tripleepoch"
    - results.json with test scores
    
    Creates a plot with:
    - X-axis: pos+neg sample count (actual_positive_samples + actual_negative_samples)
    - Y-axis: Learning rate
    - Color: Test score (recall@1fpr)
    """
    
    base_dir = Path("/workspace/GIT_SHENANIGANS/self-obfuscation/self_obfuscation_experiment/outputs/probe_checkpoints")
    target_model_path = "experiments/self_obfuscation_v1_rated/outputs/model_checkpoints/jul23_highlr_tripleepoch"
    
    data = []
    
    # Scan all folders in the probe_checkpoints directory
    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
            
        metadata_file = folder / "metadata.json"
        results_file = folder / "results.json"
        
        # Check if both files exist
        if not (metadata_file.exists() and results_file.exists()):
            print(f"Skipping {folder.name}: missing metadata.json or results.json")
            continue
            
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if this matches our target model
            if metadata.get("model_name_or_path") != target_model_path:
                print(f"Skipping {folder.name}: different model path")
                continue
                
            # Load results
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract required data
            learning_rate = metadata.get("learning_rate")
            actual_positive_samples = metadata.get("actual_positive_samples")
            actual_negative_samples = metadata.get("actual_negative_samples")
            
            # Get test score (recall@1fpr first value)
            test_recall = results.get("metrics", {}).get("test", {}).get("recall@1fpr")
            if test_recall and len(test_recall) > 0:
                test_score = test_recall[0]  # First value
            else:
                print(f"Warning: No test recall@1fpr found for {folder.name}")
                continue
            
            # Calculate total samples
            total_samples = actual_positive_samples + actual_negative_samples
            
            data.append({
                'folder': folder.name,
                'learning_rate': learning_rate,
                'total_samples': total_samples,
                'positive_samples': actual_positive_samples,
                'negative_samples': actual_negative_samples,
                'test_score': test_score
            })
            
            print(f"Found match: {folder.name} - LR: {learning_rate}, Samples: {total_samples}, Score: {test_score:.3f}")
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error processing {folder.name}: {e}")
            continue
    
    if not data:
        print("No matching folders found!")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    
    print(f"\nFound {len(df)} matching experiments")
    print(f"Learning rates: {sorted(df['learning_rate'].unique())}")
    print(f"Sample counts: {sorted(df['total_samples'].unique())}")
    print(f"Test scores range: {df['test_score'].min():.3f} - {df['test_score'].max():.3f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(df['total_samples'], df['learning_rate'], 
                         c=df['test_score'], cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Test Score (recall@1fpr)', fontsize=12)
    
    # Set labels and title
    plt.xlabel('Total Sample Count (Positive + Negative)', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Probe Performance vs Learning Rate and Sample Count\n' + 
              f'Model: {target_model_path.split("/")[-1]}', fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Use log scale for both axes
    plt.xscale('log')
    plt.xlabel('Total Sample Count (Positive + Negative) - Log Scale', fontsize=12)
    
    # Use log scale for y-axis if learning rates vary significantly
    if df['learning_rate'].max() / df['learning_rate'].min() > 10:
        plt.yscale('log')
        plt.ylabel('Learning Rate (log scale)', fontsize=12)
    
    # Annotate points with folder names (optional, might be cluttered)
    if len(df) <= 20:  # Only annotate if not too many points
        for _, row in df.iterrows():
            plt.annotate(row['folder'].split('_')[1], # Just show date part
                        (row['total_samples'], row['learning_rate']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "probe_checkpoints_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Save data as CSV for further analysis
    csv_file = "probe_checkpoints_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Data saved as: {csv_file}")
    
    plt.show()
    
    return df

if __name__ == "__main__":
    df = analyze_probe_checkpoints()