#!/usr/bin/env python3
"""
Script to generate loss plots from saved loss_data.json files.
Calculates loss10 (10-step moving average) and generates plots.

Usage:
    python plot_loss_data.py /path/to/loss_data.json
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def calculate_loss10(losses: List[float]) -> List[float]:
    """Calculate 10-step moving average of losses."""
    loss10 = []
    for i in range(len(losses)):
        if i < 9:  # Skip until window is full
            continue
        window_start = i - 9
        window_losses = losses[window_start:i + 1]
        loss10.append(np.mean(window_losses))
    return loss10


def calculate_loss50(losses: List[float]) -> List[float]:
    """Calculate 50-step moving average of losses."""
    loss50 = []
    for i in range(len(losses)):
        if i < 49:  # Skip until window is full
            continue
        window_start = i - 49
        window_losses = losses[window_start:i + 1]
        loss50.append(np.mean(window_losses))
    return loss50


def plot_loss_data(
    behavior_losses: List[float], 
    obf_losses: List[float], 
    loss10_behavior: List[float], 
    loss10_obf: List[float],
    loss50_behavior: List[float], 
    loss50_obf: List[float], 
    output_dir: str
):
    """Generate all loss plots."""
    
    # Original losses plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(behavior_losses, label='Behavior Loss', color='blue', alpha=0.7)
    plt.title('Behavior Loss Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Behavior Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(obf_losses, label='Obfuscation Loss', color='red', alpha=0.7)
    plt.title('Obfuscation Loss Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Obfuscation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    original_path = Path(output_dir) / "training_losses.png"
    plt.savefig(original_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined original losses plot
    plt.figure(figsize=(12, 6))
    
    # Normalize losses for better visualization
    norm_behavior = np.array(behavior_losses) / max(behavior_losses) if behavior_losses and max(behavior_losses) > 0 else []
    norm_obf = np.array(obf_losses) / max(obf_losses) if obf_losses and max(obf_losses) > 0 else []
    
    if len(norm_behavior) > 0:
        plt.plot(norm_behavior, label='Behavior Loss (normalized)', color='blue', alpha=0.7)
    if len(norm_obf) > 0:
        plt.plot(norm_obf, label='Obfuscation Loss (normalized)', color='red', alpha=0.7)
    
    plt.title('Training Loss Curves (Normalized)')
    plt.xlabel('Training Step')
    plt.ylabel('Normalized Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    combined_path = Path(output_dir) / "training_losses_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss10 plots
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    steps_10 = list(range(10, 10 + len(loss10_behavior)))
    plt.plot(steps_10, loss10_behavior, label='Behavior Loss10', color='blue', alpha=0.7)
    plt.title('Behavior Loss10 Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Behavior Loss10')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(steps_10, loss10_obf, label='Obfuscation Loss10', color='red', alpha=0.7)
    plt.title('Obfuscation Loss10 Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Obfuscation Loss10')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss10_path = Path(output_dir) / "training_losses_10.png"
    plt.savefig(loss10_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined loss10 plot
    plt.figure(figsize=(12, 6))
    
    # Normalize loss10 for better visualization
    norm_loss10_behavior = np.array(loss10_behavior) / max(loss10_behavior) if loss10_behavior and max(loss10_behavior) > 0 else []
    norm_loss10_obf = np.array(loss10_obf) / max(loss10_obf) if loss10_obf and max(loss10_obf) > 0 else []
    
    if len(norm_loss10_behavior) > 0:
        steps_10 = list(range(10, 10 + len(norm_loss10_behavior)))
        plt.plot(steps_10, norm_loss10_behavior, label='Behavior Loss10 (normalized)', color='blue', alpha=0.7)
    if len(norm_loss10_obf) > 0:
        steps_10 = list(range(10, 10 + len(norm_loss10_obf)))
        plt.plot(steps_10, norm_loss10_obf, label='Obfuscation Loss10 (normalized)', color='red', alpha=0.7)
    
    plt.title('Training Loss10 Curves (Normalized)')
    plt.xlabel('Training Step')
    plt.ylabel('Normalized Loss10')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    combined_loss10_path = Path(output_dir) / "training_losses_10_combined.png"
    plt.savefig(combined_loss10_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss50 plots
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    steps_50 = list(range(50, 50 + len(loss50_behavior)))
    plt.plot(steps_50, loss50_behavior, label='Behavior Loss50', color='blue', alpha=0.7)
    plt.title('Behavior Loss50 Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Behavior Loss50')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(steps_50, loss50_obf, label='Obfuscation Loss50', color='red', alpha=0.7)
    plt.title('Obfuscation Loss50 Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Obfuscation Loss50')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss50_path = Path(output_dir) / "training_losses_50.png"
    plt.savefig(loss50_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined loss50 plot
    plt.figure(figsize=(12, 6))
    
    # Normalize loss50 for better visualization
    norm_loss50_behavior = np.array(loss50_behavior) / max(loss50_behavior) if loss50_behavior and max(loss50_behavior) > 0 else []
    norm_loss50_obf = np.array(loss50_obf) / max(loss50_obf) if loss50_obf and max(loss50_obf) > 0 else []
    
    if len(norm_loss50_behavior) > 0:
        steps_50 = list(range(50, 50 + len(norm_loss50_behavior)))
        plt.plot(steps_50, norm_loss50_behavior, label='Behavior Loss50 (normalized)', color='blue', alpha=0.7)
    if len(norm_loss50_obf) > 0:
        steps_50 = list(range(50, 50 + len(norm_loss50_obf)))
        plt.plot(steps_50, norm_loss50_obf, label='Obfuscation Loss50 (normalized)', color='red', alpha=0.7)
    
    plt.title('Training Loss50 Curves (Normalized)')
    plt.xlabel('Training Step')
    plt.ylabel('Normalized Loss50')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    combined_loss50_path = Path(output_dir) / "training_losses_50_combined.png"
    plt.savefig(combined_loss50_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Original loss plots saved to {original_path} and {combined_path}")
    print(f"Loss10 plots saved to {loss10_path} and {combined_loss10_path}")
    print(f"Loss50 plots saved to {loss50_path} and {combined_loss50_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate loss plots from loss_data.json')
    parser.add_argument('loss_data_path', help='Path to loss_data.json file')
    parser.add_argument('--output_dir', '-o', help='Output directory for plots (default: same as input file)')
    
    args = parser.parse_args()
    
    # Load loss data
    loss_data_path = Path(args.loss_data_path)
    if not loss_data_path.exists():
        print(f"Error: File {loss_data_path} not found")
        return
    
    with open(loss_data_path, 'r') as f:
        loss_data = json.load(f)
    
    # Extract loss arrays
    behavior_losses = loss_data.get('behavior_losses', [])
    obf_losses = loss_data.get('obf_losses', [])
    
    # Check if loss10 data already exists
    if 'loss10_behavior' in loss_data and 'loss10_obf' in loss_data:
        loss10_behavior = loss_data['loss10_behavior']
        loss10_obf = loss_data['loss10_obf']
        print("Using existing loss10 data from file")
    else:
        # Calculate loss10
        print("Calculating loss10 (10-step moving average)...")
        loss10_behavior = calculate_loss10(behavior_losses)
        loss10_obf = calculate_loss10(obf_losses)
    
    # Check if loss50 data already exists
    if 'loss50_behavior' in loss_data and 'loss50_obf' in loss_data:
        loss50_behavior = loss_data['loss50_behavior']
        loss50_obf = loss_data['loss50_obf']
        print("Using existing loss50 data from file")
    else:
        # Calculate loss50
        print("Calculating loss50 (50-step moving average)...")
        loss50_behavior = calculate_loss50(behavior_losses)
        loss50_obf = calculate_loss50(obf_losses)
    
    # Set output directory
    output_dir = args.output_dir or loss_data_path.parent
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"Generating plots in {output_dir}...")
    plot_loss_data(behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf, output_dir)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total training steps: {len(behavior_losses)}")
    print(f"Final behavior loss: {behavior_losses[-1]:.4f}")
    print(f"Final obfuscation loss: {obf_losses[-1]:.4f}")
    print(f"Final behavior loss10: {loss10_behavior[-1]:.4f}")
    print(f"Final obfuscation loss10: {loss10_obf[-1]:.4f}")
    
    print(f"Average behavior loss: {np.mean(behavior_losses):.4f}")
    print(f"Average obfuscation loss: {np.mean(obf_losses):.4f}")
    print(f"Average behavior loss10: {np.mean(loss10_behavior):.4f}")
    print(f"Average obfuscation loss10: {np.mean(loss10_obf):.4f}")


if __name__ == "__main__":
    main()