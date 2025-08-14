#!/usr/bin/env python3
"""
Full self-obfuscation pipeline: trains models, then runs probe training/evaluation/plotting.

This script first trains a model using concept_model_train_rated.py, then calls the probe pipeline
to train probes on the resulting model, evaluate them, and produce plots.

Initially configured for quick testing with a "tiny test" model configuration.
"""

import subprocess
import sys
import os
import glob
import re
from datetime import datetime
from pathlib import Path

def run_model_training(config_name, train_args):
    """Run model training and return the checkpoint path."""
    print(f"\n{'='*60}")
    print(f"Starting model training: {config_name}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Run concept model training
    print(f"\n--- Training concept model ---")
    train_script_path = "experiments/self_obfuscation_v1_rated/concept_model_train_rated.py"
    train_cmd = [sys.executable, train_script_path] + train_args
    print(f"Training command: {' '.join(train_cmd)}")
    
    try:
        train_result = subprocess.run(train_cmd, check=True, cwd=project_root, text=True)
        print(f"✓ Model training completed successfully for {config_name}")
        
        # Find the checkpoint path - extract custom_folder_name from train_args
        checkpoint_path = None
        custom_folder_name = None
        
        # Extract custom_folder_name from training arguments
        if "--custom_folder_name" in train_args:
            idx = train_args.index("--custom_folder_name")
            if idx + 1 < len(train_args):
                custom_folder_name = train_args[idx + 1]
        
        if custom_folder_name:
            # Look for exact match with custom folder name
            output_dir = os.path.join(project_root, "experiments/self_obfuscation_pipeline/outputs/model_checkpoints")
            potential_path = os.path.join(output_dir, custom_folder_name)
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
                print(f"Found checkpoint path: {checkpoint_path}")
            else:
                print(f"Expected checkpoint path does not exist: {potential_path}")
        
        if not checkpoint_path:
            # Fallback: try to find the most recent checkpoint with any pattern
            output_dir = os.path.join(project_root, "experiments/self_obfuscation_pipeline/outputs/model_checkpoints")
            if os.path.exists(output_dir):
                # Get all subdirectories and find the most recent one
                checkpoint_dirs = [d for d in glob.glob(os.path.join(output_dir, "*")) if os.path.isdir(d)]
                if checkpoint_dirs:
                    checkpoint_path = max(checkpoint_dirs, key=os.path.getctime)
                    print(f"Using most recent checkpoint as fallback: {checkpoint_path}")
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"⚠️  Could not find model checkpoint")
            return False, None
            
        return True, checkpoint_path
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Model training failed for {config_name} with exit code {e.returncode}")
        return False, None

def run_probe_pipeline(model_checkpoint_path, quick_test_mode=True):
    """Run the probe pipeline for a trained model."""
    print(f"\n{'='*60}")
    print(f"Starting probe pipeline")
    print(f"Model checkpoint: {model_checkpoint_path}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Get project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Call the probe pipeline script
    probe_pipeline_script = "experiments/self_obfuscation_pipeline/run_probe_pipeline.py"
    probe_cmd = [sys.executable, probe_pipeline_script, model_checkpoint_path]
    
    if quick_test_mode:
        probe_cmd.append("--quick-test")
        print(f"Note: Limiting deception training samples to 50")
        probe_cmd.extend(["--deception-train-limit", "50", "--deception-test-limit", "5"])
    else:
        probe_cmd.append("--full-mode")
        print(f"Note: Limiting deception training samples to 10,000")
        probe_cmd.extend(["--deception-train-limit", "10000", "--deception-test-limit", "1000"])
    
    print(f"Probe pipeline command: {' '.join(probe_cmd)}")
    
    try:
        probe_result = subprocess.run(probe_cmd, check=True, cwd=project_root, text=True)
        print(f"✓ Probe pipeline completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Probe pipeline failed with exit code {e.returncode}")
        return False

def run_full_experiment(config_name, train_args, quick_test_mode=True):
    """Run a complete experiment: model training + probe pipeline."""
    print(f"\n{'='*80}")
    print(f"STARTING FULL EXPERIMENT: {config_name}")
    print(f"Quick test mode: {quick_test_mode}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Step 1: Train model
    success, checkpoint_path = run_model_training(config_name, train_args)
    if not success or not checkpoint_path:
        return False, None
    
    # Step 2: Run probe pipeline
    pipeline_success = run_probe_pipeline(checkpoint_path, quick_test_mode)
    
    return pipeline_success, checkpoint_path

def main():
    """Run the full pipeline with multiple predefined model configurations."""
    print("Starting full self-obfuscation pipeline...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"Project root directory: {project_root}")
    
    # Common dataset arguments
    common_dataset_args = [
        "--dataset_name", "mega_dataset_evaluated_20250729_144844_harm_batch_train.json",
        "--ultrachat_dataset_name", "ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json",
        "--max_neg_rating", "0.0"
    ]
    
    # Pipeline-specific output directory (separate from other experiments)
    pipeline_output_dir = "experiments/self_obfuscation_pipeline/outputs/model_checkpoints"
    
    # Define the two probe checkpoint paths - using original paths since these are pre-trained probes
    probe_checkpoint_4e4 = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_aug12_75_4e-5_no_gibberish"
    probe_checkpoint_4e5 = "experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_aug12_75_4e-5"
    
    # Define base model experiment configurations
    base_model_experiments = [
        {
            "name": "aug13_2_pipeline_tiny_test",
            "description": "Tiny test configuration for quick pipeline testing (5 concepts, 5 samples each)",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "4e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "5",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting",
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_8e-6_lr",
            "description": "Configuration with high LR",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "8e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "gibberish", "jokey", "literature-focused", "mathematical",
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_8e-6_lr_0.7dual",
            "description": "Configuration with high LR",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "8e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--clue_X_data_XY_probe_XY_topical", "0.7",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "gibberish", "jokey", "literature-focused", "mathematical",
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_no_gibberish_8e-6_lr",
            "description": "Baseline configuration with gibberish disabled (it's low diversity)",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "8e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "jokey", "literature-focused", "mathematical",
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_no_gibberish_8e-6_lr_3ep",
            "description": "Baseline configuration with gibberish disabled (it's low diversity)",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "3",
                "--learning_rate", "8e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "jokey", "literature-focused", "mathematical",
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_8e-6_lr_3e",
            "description": "Configuration with high LR",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "3",
                "--learning_rate", "8e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "gibberish", "jokey", "literature-focused", "mathematical",
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_10",
            "description": "Baseline configuration with higher data threshold",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "4e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "gibberish", "jokey", "literature-focused", "mathematical",
                "--max_pos_rating", "1.0"
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_baseline",
            "description": "Baseline configuration",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "4e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--filter_to_concepts", "HTML", "all-caps", "gibberish", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "jokey", "literature-focused", "mathematical",
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_no_gibberish",
            "description": "Baseline configuration with gibberish disabled (it's low diversity)",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "4e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "jokey", "literature-focused", "mathematical",
            ] + common_dataset_args,
        },
        {
            "name": "aug13_2_no_synonyms",
            "description": "Baseline configuration with synonyms disabled",
            "train_args": [
                "--validation_frequency", "750",
                "--num_epochs", "1",
                "--learning_rate", "4e-6",
                "--behaviour_lambda", "0.1",
                "--num_samples_per_concept", "450",
                "--filter_to_concepts", "HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german", "gibberish", "jokey", "literature-focused", "mathematical",
                "--no_synonyms",
            ] + common_dataset_args,
        },
    ]

    # Duplicate each model experiment for both probe checkpoints
    model_experiments = []
    for base_exp in base_model_experiments:
        base_name = base_exp["name"]
        base_description = base_exp["description"]
        base_train_args = base_exp["train_args"].copy()
        
        # Version 2: with probe_checkpoint_4e5
        exp2_train_args = base_train_args.copy()
        exp2_train_args.extend([
            "--custom_folder_name", base_name + "_4e-5probe",
            "--output_dir_base", pipeline_output_dir,
            "--probe_checkpoints", probe_checkpoint_4e5
        ])
        
        exp2 = {
            "name": base_name + "_4e-5probe",
            "description": base_description + " (using 4e-5 probe checkpoints)",
            "train_args": exp2_train_args,
        }
        model_experiments.append(exp2)
        
        if "tiny_test" not in base_name:
                    # Version 1: with probe_checkpoint_4e4
            exp1_train_args = base_train_args.copy()
            exp1_train_args.extend([
                "--custom_folder_name", base_name + "_4e-5probe_no_gibberish",
                "--output_dir_base", pipeline_output_dir,
                "--probe_checkpoints", probe_checkpoint_4e4
            ])
            
            exp1 = {
                "name": base_name + "_4e-4probe",
                "description": base_description + " (using 4e-4 probe checkpoints)",
                "train_args": exp1_train_args,
            }
            model_experiments.append(exp1)
    
    # Track results
    model_results = []
    successful_models = []
    
    # Run model training experiments
    print(f"\n{'='*80}")
    print("TRAINING MODELS")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    for i, exp_config in enumerate(model_experiments):
        config_name = exp_config["name"]
        description = exp_config["description"]
        train_args = exp_config["train_args"]
        
        print(f"\n{'-'*60}")
        print(f"Model Experiment {i+1}/{len(model_experiments)}: {config_name}")
        print(f"Description: {description}")
        print(f"{'-'*60}")
        
        success, checkpoint_path = run_model_training(config_name, train_args)
        model_results.append((config_name, success, description))
        
        if success and checkpoint_path:
            successful_models.append((config_name, checkpoint_path))
            
            # Run probe pipeline on this model
            print(f"\n{'='*80}")
            print(f"RUNNING PROBE PIPELINE ON {config_name}")
            print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            # Use quick test mode only for the first model
            quick_test_mode = (i == 0)  # Only first model uses quick test mode
            pipeline_success = run_probe_pipeline(checkpoint_path, quick_test_mode)
            
            if not pipeline_success:
                print(f"⚠️  Probe pipeline failed for {config_name}, but continuing with other models...")
        else:
            print(f"❌ Model training failed for {config_name}, skipping probe pipeline")
    
    # Summary
    print(f"\n{'='*80}")
    print("FULL PIPELINE SUMMARY")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    print(f"\nMODEL TRAINING RESULTS:")
    for config_name, success, description in model_results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {config_name}")
        print(f"  Description: {description}")
    
    if successful_models:
        print(f"\n{'='*60}")
        print("SUCCESSFUL MODEL CHECKPOINTS")
        print(f"{'='*60}")
        for config_name, checkpoint_path in successful_models:
            print(f"{config_name}:")
            print(f"  Path: {checkpoint_path}")
    
    total_experiments = len(model_results)
    successful_runs = sum(1 for _, success, _ in model_results if success)
    
    print(f"\nCompleted {successful_runs}/{total_experiments} model training experiments successfully")
    print(f"Probe pipeline was run on each successful model")
    print(f"Results saved to: experiments/self_obfuscation_pipeline/outputs/")
    
    if successful_runs > 0:
        print(f"\n🎉 Pipeline completed with {successful_runs} successful models!")
        return 0
    else:
        print(f"\n❌ Pipeline failed - no models were trained successfully")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)