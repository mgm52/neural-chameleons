#!/usr/bin/env python3
"""
Script to run concept_model_train_rated.py with 3 different configurations.

This script executes the concept model training experiment 3 times with varying parameters
to explore different training scenarios for self-obfuscation against concept probes.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_experiment(config_name, args):
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {config_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Use relative path from project root to the script
    script_path = "experiments/self_obfuscation_v1_rated/concept_model_train_rated.py"
    cmd = [sys.executable, script_path] + args
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run from project root, not from the script directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        result = subprocess.run(cmd, check=True, cwd=project_root)
        print(f"\n✓ {config_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {config_name} failed with exit code {e.returncode}")
        return False

def main():
    """Run all three experiment configurations."""
    print("Starting concept model training variations...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"Project root directory: {project_root}")
    
    # Configuration 1: Low behavior lambda (minimal behavior preservation)
    config1_args = [
        "--custom_folder_name", "jul22regular",
        "--num_samples_per_concept", "150",
        "--num_epochs", "1",
        "--learning_rate", "2e-6",
        "--validation_frequency", "500",
        "--dont_save_model"
    ]
    
    # Configuration 2: Balanced behavior lambda (moderate behavior preservation)
    config2_args = [
        "--custom_folder_name", "jul22highlr", 
        "--num_samples_per_concept", "150",
        "--num_epochs", "1",
        "--learning_rate", "4e-6",
        "--validation_frequency", "500",
        "--dont_save_model"
    ]

    config3_args = [
        "--custom_folder_name", "jul22doubleepoch",
        "--num_samples_per_concept", "150",
        "--num_epochs", "2",
        "--learning_rate", "2e-6",
        "--validation_frequency", "500",
        "--dont_save_model"
    ]

    config4_args = [
        "--custom_folder_name", "jul22_01",
        "--num_samples_per_concept", "150",
        "--num_epochs", "1",
        "--learning_rate", "2e-6",
        "--validation_frequency", "200",
        "--max_neg_rating", "0.0",
        "--min_pos_rating", "1.0",
        "--dont_save_model"
    ]
    
    config5_args = [
        "--custom_folder_name", "jul22filtered",
        "--num_samples_per_concept", "150",
        "--num_epochs", "1",
        "--learning_rate", "2e-6",
        "--validation_frequency", "200",
        "--dont_save_model",
        "--filter_to_concepts", "comforting", "therapeutic", "gibberish", "french-language", "abstract", "capitalised", "fun"
    ]
    
    experiments = [
        ("Config 1: ", config1_args),
        ("Config 2: ", config2_args), 
        ("Config 3: ", config3_args),
        ("Config 4: ", config4_args),
        ("Config 5: ", config5_args)
    ]
    
    results = []
    for config_name, args in experiments:
        success = run_experiment(config_name, args)
        results.append((config_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for config_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {config_name}")
    
    successful_runs = sum(1 for _, success in results if success)
    print(f"\nCompleted {successful_runs}/{len(results)} experiments successfully")
    
    if successful_runs == len(results):
        print("\n🎉 All experiments completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - successful_runs} experiments failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)