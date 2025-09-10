#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import sys

def plot_obf_loss50(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    steps = []
    obf_loss50_values = []
    
    for entry in data:
        if 'step' in entry and 'obf_loss50' in entry:
            steps.append(entry['step'])
            obf_loss50_values.append(entry['obf_loss50'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, obf_loss50_values, linewidth=1.5)
    plt.xlabel('Training Step')
    plt.ylabel('Obfuscation Loss (50-step moving avg)')
    plt.title('Obfuscation Loss50 Over Time')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_file = 'obf_loss50_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Show some stats
    print(f"Total training steps: {len(steps)}")
    print(f"Initial obf_loss50: {obf_loss50_values[0]:.6f}")
    print(f"Final obf_loss50: {obf_loss50_values[-1]:.6f}")
    print(f"Min obf_loss50: {min(obf_loss50_values):.6f}")
    print(f"Max obf_loss50: {max(obf_loss50_values):.6f}")
    
    plt.show()

if __name__ == "__main__":
    json_file = "experiments/self_obfuscation_v1_rated/outputs/model_checkpoints/20250721_021828_checkpoint_concepts/detailed_training_logs.json"
    plot_obf_loss50(json_file)