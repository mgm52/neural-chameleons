#!/usr/bin/env python3
"""
Automation script to process deception test result folders.

For each test result folder from 20250803_151157_deception_probe_evaluation onward:
1. Extract model checkpoint path from detailed_results.json
2. Train a new deception probe with lr=1e-3 using train_behavior_probe.py
3. Evaluate the new probe against the model checkpoint using evaluate_probe.py
4. Generate cosine similarity analysis plots using cosine_similarity_analysis.py

Requirements:
- Run this script from the repository root with .venv/bin/python
- All result folders must contain detailed_results.json with model_dir field
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def get_folders_from_timestamp(base_dir: str, start_timestamp: str) -> List[str]:
    """Get all deception probe evaluation folders from a specific timestamp onward."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return []
    
    folders = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.endswith("_deception_probe_evaluation"):
            # Extract timestamp from folder name (format: YYYYMMDD_HHMMSS_...)
            timestamp = folder.name[:15]  # Take first 15 chars: YYYYMMDD_HHMMSS
            if timestamp >= start_timestamp:
                folders.append(str(folder))
    
    return sorted(folders)


def extract_model_dir(detailed_results_path: str) -> Optional[str]:
    """Extract model_dir from detailed_results.json."""
    try:
        with open(detailed_results_path, 'r') as f:
            data = json.load(f)
            return data.get('model_dir')
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading {detailed_results_path}: {e}")
        return None


def create_probe_training_config(model_dir: str, lr: float = 1e-3) -> str:
    """Create a temporary training config file for the probe."""
    config_content = f"""
model_settings:
  model_name: "{model_dir}"

probe_settings:
  probe_type: "logistic_baseline"
  num_epochs: 10
  learning_rate: {lr}
  batch_size: 16
  split_percentage: 0.8

dataset_settings:
  dataset_name: "Mechanistic-Anomaly-Detection/gemma2-jailbreaks"
  positive_behavior: "dolus_deceptive_train" 
  negative_behavior: "circuit_breakers_train"
  split_percentage: 0.8
  limit: -1

general_settings:
  log_level: "INFO"
  device: "auto"
"""
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content.strip())
        return f.name


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\nRunning: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Success: {description}")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False


def process_result_folder(folder_path: str) -> bool:
    """Process a single result folder through the complete pipeline."""
    folder_name = Path(folder_path).name
    print(f"\n{'='*80}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*80}")
    
    # 1. Extract model directory
    detailed_results_path = os.path.join(folder_path, "detailed_results.json")
    model_dir = extract_model_dir(detailed_results_path)
    
    if not model_dir:
        print(f"Error: Could not extract model_dir from {detailed_results_path}")
        return False
    
    print(f"Model directory: {model_dir}")
    
    # 2. Create probe training config
    config_path = create_probe_training_config(model_dir, lr=1e-3)
    
    try:
        # 3. Train new deception probe
        train_cmd = [
            ".venv/bin/python", 
            "experiments/self_obfuscation_deception/train_behavior_probe.py",
            "--config", config_path
        ]
        
        if not run_command(train_cmd, f"Training probe for {folder_name}"):
            return False
        
        # 4. Find the newly created probe directory (most recent one)
        probe_base_dir = "experiments/self_obfuscation_deception/outputs/probe_checkpoints"
        probe_dirs = sorted([d for d in Path(probe_base_dir).iterdir() if d.is_dir()], 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not probe_dirs:
            print("Error: No probe directories found")
            return False
        
        newest_probe_dir = str(probe_dirs[0])
        print(f"Using newest probe directory: {newest_probe_dir}")
        
        # 5. Evaluate the probe
        eval_cmd = [
            ".venv/bin/python",
            "experiments/self_obfuscation_deception/evaluate_probe.py",
            "--probe_dir", newest_probe_dir,
            "--model_dir", model_dir,
            "--output_dir", "experiments/self_obfuscation_deception/outputs/test_results"
        ]
        
        if not run_command(eval_cmd, f"Evaluating probe for {folder_name}"):
            return False
        
        # 6. Find the newly created evaluation results directory
        results_base_dir = "experiments/self_obfuscation_deception/outputs/test_results"
        result_dirs = sorted([d for d in Path(results_base_dir).iterdir() if d.is_dir()], 
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not result_dirs:
            print("Error: No evaluation result directories found")
            return False
        
        newest_result_dir = str(result_dirs[0])
        print(f"Using newest result directory: {newest_result_dir}")
        
        # 7. Generate cosine similarity analysis plots
        # Change to the evaluation results directory so the script can find detailed_results.json
        original_cwd = os.getcwd()
        os.chdir(newest_result_dir)
        
        cosine_cmd = [
            os.path.join(original_cwd, ".venv/bin/python"),
            os.path.join(original_cwd, "experiments/self_obfuscation_deception/cosine_similarity_analysis.py")
        ]
        
        success = run_command(cosine_cmd, f"Generating cosine similarity plots for {folder_name}")
        
        # Return to original directory
        os.chdir(original_cwd)
        
        if success:
            print(f"✅ Successfully processed {folder_name}")
        else:
            print(f"❌ Failed to generate plots for {folder_name}")
        
        return success
        
    finally:
        # Clean up temporary config file
        try:
            os.unlink(config_path)
        except OSError:
            pass
    
    return False


def main():
    """Main function to process all relevant folders."""
    base_dir = "experiments/self_obfuscation_deception/outputs/test_results"
    start_timestamp = "20250803_151157"
    
    print(f"Processing deception test results from {start_timestamp} onward...")
    print(f"Base directory: {base_dir}")
    
    # Get all relevant folders
    folders = get_folders_from_timestamp(base_dir, start_timestamp)
    
    if not folders:
        print(f"No folders found from timestamp {start_timestamp} onward")
        return
    
    print(f"\nFound {len(folders)} folders to process:")
    for folder in folders:
        print(f"  - {Path(folder).name}")
    
    # Process each folder
    success_count = 0
    failed_folders = []
    
    for i, folder in enumerate(folders, 1):
        print(f"\n\nProgress: {i}/{len(folders)}")
        
        if process_result_folder(folder):
            success_count += 1
        else:
            failed_folders.append(Path(folder).name)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total folders: {len(folders)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_folders)}")
    
    if failed_folders:
        print(f"\nFailed folders:")
        for folder in failed_folders:
            print(f"  - {folder}")


if __name__ == "__main__":
    main()