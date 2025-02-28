import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

# Directory containing results
base_dir = "conditional_experiment/results"

# Lists to store detection rates and metadata
regular_rates = []
obfuscated_rates = []
absolute_effects = []
ratio_effects = []
layers = []
probe_lrs = []
model_lrs = []

# Walk through the directory to find JSON files
for root, _, files in os.walk(base_dir):
    dir_name = os.path.basename(root)
    
    # Extract metadata using regex
    layer_match = re.search(r"_l(\d+)_", dir_name)
    probe_lr_match = re.search(r"_plr_([\d.e-]+)_", dir_name)
    model_lr_match = re.search(r"_mlr_([\d.e-]+)_", dir_name)
    
    try:
        layer = int(layer_match.group(1)) if layer_match else None
        probe_lr = float(probe_lr_match.group(1)) if probe_lr_match else None
        model_lr = float(model_lr_match.group(1)) if model_lr_match else None
    except Exception as e:
        print(f"Error extracting metadata from {dir_name}: {e}")
        continue

    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            
            # Read the JSON file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Extract detection rates
                    if "averages" in data:
                        reg_rate = data["averages"].get("avg_regular_detection_rate")
                        obf_rate = data["averages"].get("avg_obfuscated_detection_rate")
                        
                        if reg_rate is not None and obf_rate is not None:
                            regular_rates.append(reg_rate)
                            obfuscated_rates.append(obf_rate)
                            absolute_effects.append(obf_rate - reg_rate)
                            ratio_effects.append(obf_rate / reg_rate if reg_rate != 0 else None)
                            layers.append(layer)
                            probe_lrs.append(probe_lr)
                            model_lrs.append(model_lr)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Function to create scatter plots with discrete layer coloring
def plot_discrete_layer_scatter(x, y, labels, title, filename):
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colors = [plt.cm.get_cmap('tab10')(i) for i in range(len(unique_labels))]
    
    for label, color in zip(unique_labels, colors):
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter([x[i] for i in indices], [y[i] for i in indices],
                    label=f"Layer {label}", color=color, alpha=0.7, edgecolors='k')
    
    plt.xlabel("Average Regular Detection Rate")
    plt.ylabel("Average Obfuscated Detection Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

# Ensure all lists have valid data
if regular_rates:
    # Plot 1A: Colored by layer with discrete legend
    plot_discrete_layer_scatter(regular_rates, obfuscated_rates, layers, 
                                f"Avg Detection Rates at 1% FPR\n(Over {len(regular_rates)} Model Runs & {8*len(regular_rates)} Probe Runs)", 
                                "regular_vs_obfuscated_by_layer.png")

    # Plot 1B: Colored by probe learning rate
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(regular_rates, obfuscated_rates, c=probe_lrs, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.colorbar(scatter, label="Probe Learning Rate")
    plt.xlabel("Average Regular Detection Rate")
    plt.ylabel("Average Obfuscated Detection Rate")
    plt.title(f"Avg Detection Rates at 1% FPR\n(Over {len(regular_rates)} Model Runs & {8*len(regular_rates)} Probe Runs)")
    plt.grid(True)
    plt.savefig("regular_vs_obfuscated_by_probe_lr.png")
    plt.close()
    print("Saved plot to regular_vs_obfuscated_by_probe_lr.png")

    # Plot 1C: Colored by model learning rate
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(regular_rates, obfuscated_rates, c=model_lrs, cmap='viridis', alpha=0.7, edgecolors='k')
    plt.colorbar(scatter, label="Model Learning Rate")
    plt.xlabel("Average Regular Detection Rate")
    plt.ylabel("Average Obfuscated Detection Rate")
    plt.title(f"Avg Detection Rates at 1% FPR\n(Over {len(regular_rates)} Model Runs & {8*len(regular_rates)} Probe Runs)")
    plt.grid(True)
    plt.savefig("regular_vs_obfuscated_by_model_lr.png")
    plt.close()
    print("Saved plot to regular_vs_obfuscated_by_model_lr.png")

    # Plot 4: Detection Rate vs Obfuscation Effect (Absolute)
    plt.figure(figsize=(8, 6))
    plt.scatter(regular_rates, absolute_effects, alpha=0.7, edgecolors='k')
    plt.xlabel("Average Regular Detection Rate")
    plt.ylabel("Obfuscation Effect (Absolute)")
    plt.title(f"Avg Detection Rates at 1% FPR\n(Over {len(regular_rates)} Model Runs & {8*len(regular_rates)} Probe Runs)")
    plt.grid(True)
    plt.savefig("detection_rate_vs_obfuscation_effect_absolute.png")
    plt.close()
    print("Saved plot to detection_rate_vs_obfuscation_effect_absolute.png")

    # Plot 5: Detection Rate vs Obfuscation Effect (Ratio)
    valid_ratio_indices = [i for i, r in enumerate(ratio_effects) if r is not None]
    valid_regular_rates = [regular_rates[i] for i in valid_ratio_indices]
    valid_ratio_effects = [ratio_effects[i] for i in valid_ratio_indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(valid_regular_rates, valid_ratio_effects, alpha=0.7, edgecolors='k')
    plt.xlabel("Average Regular Detection Rate")
    plt.ylabel("Obfuscation Effect (Ratio)")
    plt.title(f"Avg Detection Rates at 1% FPR\n(Over {len(regular_rates)} Model Runs & {8*len(regular_rates)} Probe Runs) (Ratio)")
    plt.grid(True)
    plt.savefig("detection_rate_vs_obfuscation_effect_ratio.png")
    plt.close()
    print("Saved plot to detection_rate_vs_obfuscation_effect_ratio.png")
else:
    print("No valid data found. Check if JSON files contain correct structure and directory names are parsed correctly.")
