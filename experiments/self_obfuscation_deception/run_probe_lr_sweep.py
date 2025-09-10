#!/usr/bin/env python3
"""
Script to run behavior probe training with multiple learning rates.

This script generates config files with different learning rates and runs
the train_behavior_probe.py script for each configuration.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import List


def print_flush(*args, **kwargs):
    """Print with immediate flush to ensure output appears right away."""
    print(*args, **kwargs)
    sys.stdout.flush()


def create_config_with_params(base_config_path: str, learning_rate: float = None, max_samples: int = None, max_test_samples: int = None, output_path: str = None):
    """Create a config file with the specified parameters."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update learning rate for all probes
    if learning_rate is not None:
        for probe_name, probe_config in config['probes'].items():
            if probe_config['type'] in ['logistic', 'mlp']:
                probe_config['learning_rate'] = learning_rate
    
    # Update max_samples
    if max_samples is not None:
        config['max_samples'] = max_samples
        # Also set it for individual probes if they don't have it
        for probe_name, probe_config in config['probes'].items():
            if 'max_samples' not in probe_config:
                probe_config['max_samples'] = max_samples
    
    # Update max_test_samples
    if max_test_samples is not None:
        config['max_test_samples'] = max_test_samples
        # Also set it for individual probes if they don't have it
        for probe_name, probe_config in config['probes'].items():
            if 'max_test_samples' not in probe_config:
                probe_config['max_test_samples'] = max_test_samples
    
    # Save the modified config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return output_path


def run_training_with_config(config_path: str, script_path: str):
    """Run the training script with the given config."""
    cmd = ['python', script_path, '--config', config_path]
    
    print_flush(f"Running: {' '.join(cmd)}")
    
    # Run with real-time output streaming
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        print_flush(f"Error: Training failed with return code {result.returncode}")
        return False
    else:
        print_flush(f"Successfully completed training with config {config_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run probe training with multiple learning rates and sample sizes")
    parser.add_argument("--base_config", type=str, required=True,
                       help="Path to base configuration file")
    parser.add_argument("--learning_rates", type=float, nargs='+',
                       #default=[5e-6, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 5e-4],
                       #default=[2e-4, 1e-3, 2e-3, 4e-3],
                       default=[1e-3],
                       help="Learning rates to test")
    parser.add_argument("--sample_sizes", type=int, nargs='*',
                       #default=[100, 1000, None],
                       default=[None],
                       help="Sample sizes to test (default: 100, 1000, all data)")
    parser.add_argument("--test_sample_sizes", type=int, nargs='*',
                       default=[1000],
                       help="Test sample sizes to use for evaluation (default: 1000)")
    parser.add_argument("--script_path", type=str,
                       default="experiments/self_obfuscation_deception/train_behavior_probe.py",
                       help="Path to the training script")
    parser.add_argument("--keep_configs", action="store_true",
                       help="Keep generated config files instead of deleting them")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.base_config):
        print_flush(f"Error: Base config file {args.base_config} does not exist")
        return 1
    
    if not os.path.exists(args.script_path):
        print_flush(f"Error: Training script {args.script_path} does not exist")
        return 1
    
    print_flush(f"Running probe training with learning rates: {args.learning_rates}")
    print_flush(f"Training sample sizes: {args.sample_sizes}")
    
    # Use test_sample_sizes (defaults to [1000])
    test_sample_sizes = args.test_sample_sizes
    print_flush(f"Test sample sizes: {test_sample_sizes}")
    
    print_flush(f"Base config: {args.base_config}")
    print_flush(f"Training script: {args.script_path}")
    
    successful_runs = 0
    failed_runs = 0
    temp_configs = []
    
    # Create list of all parameter combinations
    param_combinations = []
    sample_sizes = args.sample_sizes
    
    for train_samples in sample_sizes:
        # TODO: make test_sample_sizes an int, not a list.... this loop doesnt make sense lol
        for test_samples in test_sample_sizes:
            for lr in args.learning_rates:
                param_combinations.append((lr, train_samples, test_samples))
    
    print_flush(f"Total combinations to test: {len(param_combinations)}")
    
    try:
        for i, (lr, train_samples, test_samples) in enumerate(param_combinations):
            print_flush(f"\n{'='*60}")
            print_flush(f"Combination {i+1}/{len(param_combinations)}")
            print_flush(f"Learning rate: {lr}")
            print_flush(f"Training samples: {train_samples if train_samples else 'all data'}")
            print_flush(f"Test samples: {test_samples if test_samples else 'all data'}")
            print_flush(f"{'='*60}")
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                temp_config_path = f.name
                temp_configs.append(temp_config_path)
            
            # Generate config with current parameters
            create_config_with_params(
                base_config_path=args.base_config,
                learning_rate=lr,
                max_samples=train_samples,
                max_test_samples=test_samples,
                output_path=temp_config_path
            )
            
            # Run training
            success = run_training_with_config(temp_config_path, args.script_path)
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
                print_flush(f"Failed to train with lr={lr}, train_samples={train_samples}, test_samples={test_samples}")
    
    finally:
        # Clean up temporary config files unless requested to keep them
        if not args.keep_configs:
            for temp_config in temp_configs:
                try:
                    os.unlink(temp_config)
                except OSError:
                    pass
        else:
            print_flush(f"\nGenerated config files saved:")
            for i, temp_config in enumerate(temp_configs):
                lr, train_samples, test_samples = param_combinations[i]
                saved_path = f"config_lr_{lr}_train_{train_samples if train_samples else 'all'}_test_{test_samples if test_samples else 'all'}.yaml"
                os.rename(temp_config, saved_path)
                print_flush(f"  lr={lr}, train={train_samples}, test={test_samples}: {saved_path}")
    
    print_flush(f"\n{'='*60}")
    print_flush(f"FINAL SUMMARY")
    print_flush(f"{'='*60}")
    print_flush(f"Successful runs: {successful_runs}")
    print_flush(f"Failed runs: {failed_runs}")
    print_flush(f"Total runs: {successful_runs + failed_runs}")
    
    return 0 if failed_runs == 0 else 1


if __name__ == "__main__":
    exit(main())