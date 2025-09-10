"""
Script to generate loss10 and loss50 plots from harmful model training loss data.

Reads loss_data.json from a checkpoint directory and generates moving average plots.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path


def moving_average(data, window_size):
    """Calculate moving average with specified window size."""
    if len(data) < window_size:
        return [], []
    
    # Calculate moving average starting only when window is complete
    ma_values = []
    for i in range(window_size - 1, len(data)):
        window = data[i - window_size + 1:i + 1]
        ma_values.append(np.mean(window))
    
    # Return both the moving average values and the corresponding x indices
    x_indices = list(range(window_size - 1, len(data)))
    return ma_values, x_indices


def plot_losses_with_moving_average(
    behavior_losses, 
    obf_losses, 
    window_size, 
    save_path, 
    title_suffix=""
):
    """Generate loss plots with moving average."""
    
    # Calculate moving averages
    behavior_ma, behavior_x = moving_average(behavior_losses, window_size)
    obf_ma, obf_x = moving_average(obf_losses, window_size)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Behavior losses
    ax1.plot(behavior_x, behavior_ma, color='blue', linewidth=2)
    ax1.set_title(f'Behavior Loss{title_suffix}')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Obfuscation losses
    ax2.plot(obf_x, obf_ma, color='red', linewidth=2)
    ax2.set_title(f'Obfuscation Loss{title_suffix}')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate loss plots from harmful model training data')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='experiments/self_obfuscation_harmful_2/outputs/model_checkpoints/checkpoint_20250704_081933_harmful',
        help='Path to checkpoint directory containing loss_data.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (defaults to checkpoint_dir)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    checkpoint_dir = Path(args.checkpoint_dir)
    loss_data_path = checkpoint_dir / 'loss_data.json'
    
    if not loss_data_path.exists():
        raise FileNotFoundError(f"Loss data file not found: {loss_data_path}")
    
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load loss data
    print(f"Loading loss data from: {loss_data_path}")
    with open(loss_data_path, 'r') as f:
        loss_data = json.load(f)
    
    behavior_losses = loss_data.get('behavior_losses', [])
    obf_losses = loss_data.get('obf_losses', [])
    
    print(f"Loaded {len(behavior_losses)} behavior loss points and {len(obf_losses)} obfuscation loss points")
    
    # Print some summary statistics
    print(f"\nLoss Statistics:")
    print(f"  Behavior Loss - Mean: {np.mean(behavior_losses):.4f}, Std: {np.std(behavior_losses):.4f}")
    print(f"  Obfuscation Loss - Mean: {np.mean(obf_losses):.4f}, Std: {np.std(obf_losses):.4f}")
    print(f"  Max Behavior Loss: {max(behavior_losses):.4f}")
    print(f"  Max Obfuscation Loss: {max(obf_losses):.4f}")
    
    # Generate loss10 plot (10-step moving average)
    loss10_path = output_dir / 'loss10.png'
    plot_losses_with_moving_average(
        behavior_losses=behavior_losses,
        obf_losses=obf_losses,
        window_size=10,
        save_path=loss10_path,
        title_suffix=" (10-step Moving Average)"
    )
    print(f"loss10 saved to {loss10_path}")
    
    # Generate loss50 plot (50-step moving average)
    loss50_path = output_dir / 'loss50.png'
    plot_losses_with_moving_average(
        behavior_losses=behavior_losses,
        obf_losses=obf_losses,
        window_size=50,
        save_path=loss50_path,
        title_suffix=" (50-step Moving Average)"
    )
    print(f"loss50 saved to {loss50_path}")


if __name__ == "__main__":
    main()