#!/usr/bin/env python3
"""
Probe pipeline script: trains probes on a given model, evaluates them, and produces plots.
Operates on a single input model checkpoint without training new models.

This enhanced version performs:
1. Test against pre-existing deception probe and plot cosine similarity
2. Train a new deception probe targeting that checkpoint, evaluate, and plot cosine similarity
3. Test against pre-existing harmfulness probes and plot results
4. Train new harmfulness probes, evaluate them, and plot results
5. Upload all plots to wandb
"""

import argparse
import subprocess
import sys
import os
import json
import glob
import re
from datetime import datetime
from pathlib import Path
import shutil
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_latest_probe_checkpoint_dir(base_dir):
    """Find the most recently created probe checkpoint directory."""
    if not os.path.exists(base_dir):
        return None
    
    # Look for directories with metadata.json (indicating valid checkpoint)
    checkpoint_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "metadata.json")):
            checkpoint_dirs.append((item, os.path.getctime(item_path)))
    
    if not checkpoint_dirs:
        return None
    
    # Return the most recently created directory
    latest_dir = max(checkpoint_dirs, key=lambda x: x[1])[0]
    return os.path.join(base_dir, latest_dir)

def get_latest_evaluation_results_dir(base_dir):
    """Find the most recently created evaluation results directory."""
    if not os.path.exists(base_dir):
        return None
    
    # Look for timestamped directories
    eval_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and "_probe_evaluation" in item:
            # Check if it has expected result files
            if (os.path.exists(os.path.join(item_path, "detailed_results.json")) or 
                os.path.exists(os.path.join(item_path, "concept_probe_results.csv"))):
                eval_dirs.append((item, os.path.getctime(item_path)))
    
    if not eval_dirs:
        return None
    
    # Return the most recent directory
    latest_dir = max(eval_dirs, key=lambda x: x[1])[0]
    return os.path.join(base_dir, latest_dir)

def load_wandb_info_from_model_checkpoint(model_dir):
    """Load wandb run information from model checkpoint's metadata."""
    if not model_dir:
        return None
    
    metadata_path = Path(model_dir) / "model_training_metadata.json"
    if not metadata_path.exists():
        print(f"No model_training_metadata.json found in {model_dir}")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        wandb_info = metadata.get('wandb')
        if wandb_info:
            print(f"Found wandb run info: {wandb_info.get('run_name')} ({wandb_info.get('run_id')})")
            return wandb_info
        else:
            print("No wandb information found in model metadata")
            return None
    except Exception as e:
        print(f"Error reading model metadata: {e}")
        return None

def upload_plots_to_wandb(plot_paths, wandb_info, plot_type="probe_evaluation"):
    """Upload plot files to the specified wandb run (both PNG and PDF)."""
    if not wandb_info or not plot_paths:
        print("Skipping wandb upload - no wandb info or no plots")
        return
    
    try:
        import wandb
        
        # Resume the existing run
        run = wandb.init(
            project=wandb_info.get('project'),
            id=wandb_info.get('run_id'),
            resume="must",
            dir="/workspace/wandb"
        )
        
        print(f"Resumed wandb run: {wandb_info.get('run_name')} ({wandb_info.get('run_id')})")
        
        # Upload each plot (both PNG and PDF if they exist)
        for plot_path in plot_paths:
            plot_path = Path(plot_path)
            if plot_path.exists():
                # Create a more descriptive name for the artifact
                plot_name = plot_path.name
                artifact_name = f"{plot_type}_{plot_name}"
                
                # Upload the file
                wandb.save(str(plot_path), base_path=str(plot_path.parent))
                print(f"Uploaded {plot_name} to wandb as {artifact_name}")
                
                # Also check for PDF version
                pdf_path = plot_path.with_suffix('.pdf')
                if pdf_path.exists():
                    pdf_name = pdf_path.name
                    pdf_artifact_name = f"{plot_type}_{pdf_name}"
                    wandb.save(str(pdf_path), base_path=str(pdf_path.parent))
                    print(f"Uploaded {pdf_name} to wandb as {pdf_artifact_name}")
            else:
                print(f"Plot file not found: {plot_path}")
        
        wandb.finish()
        print("Finished uploading plots to wandb")
        
    except ImportError:
        print("wandb not available - skipping plot upload")
    except Exception as e:
        print(f"Error uploading plots to wandb: {e}")

def run_deception_evaluation(model_checkpoint, probe_dir, output_dir_suffix="", project_root=None, test_limit=100):
    """Run deception evaluation on the given model checkpoint."""
    print(f"\n{'='*60}")
    print(f"Starting deception evaluation")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Probe dir: {probe_dir}")
    print(f"Test limit: {test_limit}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Run deception evaluation
    print(f"\n--- Running deception evaluation ---")
    eval_script_path = "experiments/self_obfuscation_deception/evaluate_probe.py"
    
    # Build evaluation command with the checkpoint path and pipeline output directory
    pipeline_eval_output_dir = f"experiments/self_obfuscation_pipeline/outputs/evaluation_results{output_dir_suffix}"
    os.makedirs(os.path.join(project_root, pipeline_eval_output_dir), exist_ok=True)
    
    eval_args = [
        "--limit", str(test_limit),
        "--config", "configs/self_obfuscation/probe_deception_full.yaml",
        "--probe_dir", probe_dir
    ]
    eval_cmd = [sys.executable, eval_script_path, "--model_dir", model_checkpoint, "--output_dir", pipeline_eval_output_dir] + eval_args
    print(f"Evaluation command: {' '.join(eval_cmd)}")
    
    plot_paths = []
    try:
        eval_result = subprocess.run(eval_cmd, check=True, cwd=project_root, text=True)
        print(f"✓ Deception evaluation completed successfully")
        
        # Run cosine similarity analysis
        print(f"\n--- Running cosine similarity analysis ---")
        cosine_script_path = "experiments/self_obfuscation_deception/cosine_similarity_analysis.py"
        cosine_cmd = [sys.executable, cosine_script_path, "--results_dir", pipeline_eval_output_dir]
        print(f"Cosine similarity command: {' '.join(cosine_cmd)}")
        
        try:
            cosine_result = subprocess.run(cosine_cmd, check=True, cwd=project_root, text=True)
            print(f"✓ Cosine similarity analysis completed successfully")
            
            # Find generated plots (both PNG and PDF)
            for plot_base in ["cosine_similarity_plot", "deception_cosine_plot", "truth_cosine_plot"]:
                for ext in [".png", ".pdf"]:
                    plot_file = plot_base + ext
                    plot_path = os.path.join(project_root, pipeline_eval_output_dir, plot_file)
                    if os.path.exists(plot_path):
                        plot_paths.append(plot_path)
                        print(f"Found plot: {plot_file}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Cosine similarity analysis failed with exit code {e.returncode}")
            # Don't fail the overall experiment if cosine analysis fails
            
        return True, plot_paths
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Deception evaluation failed with exit code {e.returncode}")
        return False, plot_paths

def train_deception_probe(model_checkpoint, output_dir_suffix="", project_root=None, max_train_samples=None, max_test_samples=None):
    """Train a new deception probe on the given model checkpoint."""
    print(f"\n{'='*60}")
    print(f"Training new deception probe")
    print(f"Model checkpoint: {model_checkpoint}")
    if max_train_samples:
        print(f"Max training samples: {max_train_samples}")
    if max_test_samples:
        print(f"Max test samples: {max_test_samples}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Create output directory for this specific probe training
    pipeline_probe_output_dir = f"experiments/self_obfuscation_pipeline/outputs/probe_checkpoints{output_dir_suffix}"
    os.makedirs(os.path.join(project_root, pipeline_probe_output_dir), exist_ok=True)
    
    # Train deception probe
    print(f"\n--- Training deception probe ---")
    train_script_path = "experiments/self_obfuscation_deception/train_behavior_probe.py"
    
    # Build training command 
    train_cmd = [
        sys.executable, train_script_path,
        "--config", "configs/self_obfuscation/probe_deception_full.yaml"
    ]
    
    # If model_checkpoint is provided, update the config to use it
    if model_checkpoint and os.path.exists(model_checkpoint):
        # We need to create a temporary config file that uses the model checkpoint
        config_path = os.path.join(project_root, "configs/self_obfuscation/probe_deception_full.yaml")
        temp_config_path = os.path.join(project_root, pipeline_probe_output_dir, "temp_probe_config.yaml")
        
        # Read original config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update model path
        config['model_name_or_path'] = model_checkpoint
        
        # Add sample limits if specified
        if max_train_samples:
            config['max_samples'] = max_train_samples
        if max_test_samples:
            config['max_test_samples'] = max_test_samples
        
        # Write temporary config
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        train_cmd = [
            sys.executable, train_script_path,
            "--config", temp_config_path
        ]
    
    print(f"Training command: {' '.join(train_cmd)}")
    
    try:
        # Capture output to find the checkpoint directory
        train_result = subprocess.run(train_cmd, check=True, cwd=project_root, text=True, capture_output=True)
        output = train_result.stdout
        
        print(f"✓ Deception probe training completed successfully")
        
        # Extract the probe checkpoint directory from output
        # Looking for pattern like "Saved logistic_baseline probe to experiments/.../probe.pkl"
        import re
        match = re.search(r"Saved .+ probe to (.+)/probe\.pkl", output)
        if match:
            probe_dir = match.group(1)
            print(f"Found probe checkpoint directory: {probe_dir}")
            return True, probe_dir
        else:
            # Try to find the latest probe checkpoint
            probe_base = "experiments/self_obfuscation_deception/outputs/probe_checkpoints"
            latest_probe = get_latest_probe_checkpoint_dir(os.path.join(project_root, probe_base))
            if latest_probe:
                print(f"Using latest probe checkpoint: {latest_probe}")
                return True, latest_probe
            else:
                print("⚠️  Could not find probe checkpoint directory")
                return False, None
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Deception probe training failed with exit code {e.returncode}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False, None


def run_probe_harmful_experiment(model_checkpoint, training_probe_lr, is_first_run=False, output_dir_suffix="", project_root=None):
    """Run probe training and evaluation for harmful dataset."""
    print(f"\n{'='*60}")
    print(f"Starting harmful probe experiment")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Probe learning rate: {training_probe_lr}")
    print(f"First run (quick mode): {is_first_run}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Configure parameters based on first run (quick test mode)
    if is_first_run:
        print("=== RUNNING IN QUICK TEST MODE ===")
        max_samples_per_class = 20
        min_test_samples = 2
        max_test_samples = 3
        bootstrap_iterations = 100
        include_adjectives = ["toxic", "academic"]
        print(f"Parameters: max_samples_per_class={max_samples_per_class}, min_test_samples={min_test_samples}, max_test_samples={max_test_samples}, bootstrap_iterations={bootstrap_iterations}")
        print(f"Include adjectives: {include_adjectives}")
    else:
        print("=== RUNNING IN FULL MODE ===")
        max_samples_per_class = 2048
        min_test_samples = 10
        max_test_samples = 200
        bootstrap_iterations = 1000
        include_adjectives = None  # No need to specify since default is now empty
        print(f"Parameters: max_samples_per_class={max_samples_per_class}, min_test_samples={min_test_samples}, max_test_samples={max_test_samples}, bootstrap_iterations={bootstrap_iterations}")
        print("Using all adjectives found in the dataset (no include_adjectives filter)")
    
    # Train probes
    print(f"=== TRAINING PROBES (lr {training_probe_lr}) (checkpoint {model_checkpoint}) ===")
    pipeline_probe_output_dir = f"experiments/self_obfuscation_pipeline/outputs/probe_checkpoints{output_dir_suffix}"
    train_cmd = [
        sys.executable,
        "experiments/self_obfuscation_v1_rated/probe_train_synthetic_rated.py",
        "--dataset_path", "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TRAIN.json",
        "--output_dir_base", pipeline_probe_output_dir,
        "--lr", training_probe_lr,
        "--num_test_samples", "0",  # Don't hold out any test data
        "--max_samples_per_class", str(max_samples_per_class),
        "--model_checkpoint", model_checkpoint
    ]
    
    # Only add include_adjectives if there are specific adjectives to include
    if include_adjectives is not None and include_adjectives:
        train_cmd.extend(["--include_adjectives"] + include_adjectives)
    
    plot_paths = []
    try:
        train_result = subprocess.run(train_cmd, check=True, cwd=project_root, text=True)
        print("=== TRAINING COMPLETED SUCCESSFULLY ===")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code {e.returncode}")
        return False, plot_paths
    
    # Find the most recent probe checkpoint directory
    probe_checkpoint_base = os.path.join(project_root, pipeline_probe_output_dir)
    latest_probe_dir = get_latest_probe_checkpoint_dir(probe_checkpoint_base)
    
    if not latest_probe_dir:
        print("ERROR: Could not find probe checkpoint directory after training")
        return False, plot_paths
    
    print(f"Found latest probe checkpoint directory: {latest_probe_dir}")
    
    # Evaluate probes
    print(f"=== EVALUATING PROBES ({model_checkpoint}) ===")
    test_dataset_path = "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TEST.json"
    
    # Check if the test dataset exists
    full_test_path = os.path.join(project_root, test_dataset_path)
    if not os.path.exists(full_test_path):
        print(f"ERROR: Test dataset not found at {full_test_path}")
        return False, plot_paths
    
    pipeline_eval_results_dir = f"experiments/self_obfuscation_pipeline/outputs/evaluation_results{output_dir_suffix}"
    eval_cmd = [
        sys.executable,
        "experiments/self_obfuscation_v1_rated/probe_test_synthetic_clued_labelled.py",
        "--probe_dir", latest_probe_dir,
        "--model_checkpoint", model_checkpoint,
        "--dataset_file_path", test_dataset_path,
        "--output_dir", pipeline_eval_results_dir,
        "--ultrachat_mega_dataset_path", "",  # No ultrachat data as requested
        "--batch_size", "32",
        "--bootstrap_iterations", str(bootstrap_iterations),
        "--positive_threshold", "1.0",
        "--negative_threshold", "0.0",
        "--use_topical_ratings",
        "--min_test_samples", str(min_test_samples),
        "--max_test_samples", str(max_test_samples),
        "--seed", "42"
    ]
    
    try:
        eval_result = subprocess.run(eval_cmd, check=True, cwd=project_root, text=True)
        print("=== EVALUATION COMPLETED SUCCESSFULLY ===")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with return code {e.returncode}")
        return False, plot_paths
    
    # Find the latest evaluation results directory
    evaluation_results_base = os.path.join(project_root, pipeline_eval_results_dir)
    latest_eval_dir = get_latest_evaluation_results_dir(evaluation_results_base)
    
    if not latest_eval_dir:
        print("WARNING: Could not find evaluation results directory")
        print(f"Probe checkpoints saved to: {latest_probe_dir}")
        print(f"Evaluation results saved to: {evaluation_results_base}")
        return True, plot_paths  # Still consider successful if evaluation ran
    
    # Clean and format results
    print("=== CLEANING AND FORMATTING RESULTS ===")
    print(f"Found latest evaluation results directory: {latest_eval_dir}")
    
    results_cleaner_cmd = [
        sys.executable,
        "experiments/self_obfuscation_v1_rated/results_cleaner.py",
        "--results_dir", latest_eval_dir
    ]
    
    try:
        cleaner_result = subprocess.run(results_cleaner_cmd, check=True, cwd=project_root, text=True)
        print("=== RESULTS CLEANING COMPLETED SUCCESSFULLY ===")
        
        # Check for generated plots (both PNG and PDF)
        for ext in [".png", ".pdf"]:
            plot_file = os.path.join(latest_eval_dir, f"results_plot{ext}")
            if os.path.exists(plot_file):
                plot_paths.append(plot_file)
                print(f"Found plot: results_plot{ext}")
            
    except subprocess.CalledProcessError as e:
        print(f"Results cleaning failed with return code {e.returncode}")
        print("Results are still available in raw format")
    
    print(f"Probe checkpoints saved to: {latest_probe_dir}")
    print(f"Evaluation results saved to: {latest_eval_dir}")
    print(f"Formatted results (if successful): {latest_eval_dir}/results_table.txt and results_plot.png")
    
    return True, plot_paths

def train_merged_probe(model_checkpoint, merge_concepts, output_dir_suffix="", project_root=None):
    """Train a merged probe that combines multiple concepts with default learning rate."""
    return train_merged_probe_with_lr(model_checkpoint, merge_concepts, "1e-3", output_dir_suffix, project_root)

def train_merged_probe_with_lr(model_checkpoint, merge_concepts, learning_rate, output_dir_suffix="", project_root=None, quick_mode=False):
    """Train a merged probe that combines multiple concepts with specified learning rate."""
    print(f"\n{'*'*60}")
    print(f"TRAINING MERGED PROBE")
    print(f"{'*'*60}")
    print(f"Concepts to merge: {merge_concepts}")
    print(f"Number of concepts: {len(merge_concepts)}")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Learning rate: {learning_rate}")
    
    if project_root:
        os.chdir(project_root)
    
    pipeline_probe_dir = f"experiments/self_obfuscation_pipeline/outputs/probe_checkpoints{output_dir_suffix}"
    os.makedirs(pipeline_probe_dir, exist_ok=True)
    
    # Build command
    train_cmd = [
        sys.executable,
        "experiments/self_obfuscation_v1_rated/probe_train_synthetic_rated.py",
        "--dataset_path", "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TRAIN.json",
        "--output_dir_base", pipeline_probe_dir,
        "--model_checkpoint", model_checkpoint,
        "--target_layers", "12",
        "--probe_type", "logistic",
        "--lr", learning_rate,
        "--batch_size", "16",
        "--epochs", "1",
        "--positive_threshold", "0.75",
        "--negative_threshold", "0.0",
        "--use_topical_ratings",
        "--min_samples_per_class", "2" if quick_mode else "50",
        "--max_samples_per_class", "16" if quick_mode else "2048",
        "--merge_concepts_into_one_probe"
    ] + merge_concepts
    
    print(f"\nExecuting merged probe training...")
    print(f"Command: python {' '.join(train_cmd[1:])}")
    print(f"Working directory: {project_root}")
    print("-"*60)
    
    try:
        result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=project_root)
        if result.returncode != 0:
            print(f"\n✗ MERGED PROBE TRAINING FAILED")
            print(f"Return code: {result.returncode}")
            print(f"\nERROR OUTPUT:")
            print("-"*60)
            if result.stderr:
                print(result.stderr[:2000])  # Limit error output
            print("-"*60)
            print(f"\nSTANDARD OUTPUT (last 1000 chars):")
            print("-"*60)
            if result.stdout:
                print(result.stdout[-1000:])
            print("-"*60)
            return False, None
        
        print(f"\n✓ Merged probe training completed (exit code: 0)")
        
        # Extract and display key information from output
        if result.stdout:
            # Look for saved probe path
            for line in result.stdout.split('\n'):
                if "Saved" in line and "probe to" in line:
                    print(f"✓ {line.strip()}")
                elif "ERROR" in line or "WARNING" in line:
                    print(f"\u26A0 {line.strip()}")
                elif "Total positive samples:" in line or "Total negative samples:" in line:
                    print(f"  {line.strip()}")
        
        # Find the newly created probe directory
        print(f"\nSearching for probe checkpoint...")
        abs_pipeline_probe_dir = os.path.join(project_root, pipeline_probe_dir) if not os.path.isabs(pipeline_probe_dir) else pipeline_probe_dir
        latest_probe_dir = get_latest_probe_checkpoint_dir(abs_pipeline_probe_dir)
        
        if latest_probe_dir:
            print(f"✓ Found probe checkpoint: {latest_probe_dir}")
            # Check for expected files
            expected_files = ["merged_probe.pkl", "metadata.json"]
            for fname in expected_files:
                fpath = os.path.join(latest_probe_dir, fname)
                if os.path.exists(fpath):
                    fsize = os.path.getsize(fpath) / 1024  # KB
                    print(f"  ✓ {fname}: {fsize:.1f} KB")
                else:
                    print(f"  ✗ {fname}: NOT FOUND")
            return True, latest_probe_dir
        else:
            print("⚠ Could not find probe checkpoint directory in standard location")
            print(f"  Searching recursively in: {abs_pipeline_probe_dir}")
            
            # Try to find it recursively
            found_dirs = []
            for root, dirs, files in os.walk(abs_pipeline_probe_dir):
                if "merged_probe.pkl" in files or "metadata.json" in files:
                    found_dirs.append(root)
            
            if found_dirs:
                # Use the most recent one
                latest = max(found_dirs, key=lambda x: os.path.getctime(x))
                print(f"  ✓ Found probe in: {latest}")
                return True, latest
            
            print(f"  ✗ No probe files found in any subdirectory")
            return True, None
            
    except Exception as e:
        print(f"Error during merged probe training: {e}")
        return False, None

def test_merged_probe(model_checkpoint, probe_dir, trigger_words, output_dir_suffix="", project_root=None, quick_mode=False):
    """Test a merged probe using the new evaluation script compatible with cosine similarity analysis."""
    print(f"Testing merged probe with {len(trigger_words)} trigger words")
    print(f"Probe directory: {probe_dir}")
    print(f"Model checkpoint: {model_checkpoint}")
    
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    pipeline_eval_results_dir = f"experiments/self_obfuscation_pipeline/outputs/evaluation_results{output_dir_suffix}"
    os.makedirs(os.path.join(project_root, pipeline_eval_results_dir), exist_ok=True)
    
    test_dataset_path = "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TEST.json"
    
    # Build command using the new evaluation script (now in correct location)
    eval_cmd = [
        sys.executable,
        "experiments/self_obfuscation_v1_rated/evaluate_merged_probe.py",
        "--probe_dir", probe_dir,
        "--model_dir", model_checkpoint,
        "--test_dataset", test_dataset_path,
        "--output_dir", pipeline_eval_results_dir,
        "--limit", "16" if quick_mode else "-1",  # Use all data
        "--trigger_words"
    ] + trigger_words
    
    print(f"Running merged probe evaluation")
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=project_root)
        if result.returncode != 0:
            print(f"Merged probe evaluation failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, []
        
        print("Merged probe evaluation completed successfully")
        
        # Find the evaluation results directory
        abs_pipeline_eval_results_dir = os.path.join(project_root, pipeline_eval_results_dir)
        latest_eval_dir = get_latest_evaluation_results_dir(abs_pipeline_eval_results_dir)
        plot_paths = []
        
        if latest_eval_dir:
            # Run cosine similarity analysis on the results
            print(f"Running cosine similarity analysis on results...")
            cosine_cmd = [
                sys.executable,
                "experiments/self_obfuscation_deception/cosine_similarity_analysis.py",
                "--results_dir", latest_eval_dir
            ]
            
            try:
                cosine_result = subprocess.run(cosine_cmd, capture_output=True, text=True, cwd=project_root)
                if cosine_result.returncode == 0:
                    print("Cosine similarity analysis completed successfully")
                else:
                    print(f"Cosine similarity analysis failed: {cosine_result.stderr}")
            except Exception as e:
                print(f"Error running cosine similarity analysis: {e}")
            
            # Find generated plots (both PNG and PDF)
            for file in os.listdir(latest_eval_dir):
                if file.endswith('.png') or file.endswith('.pdf'):
                    plot_paths.append(os.path.join(latest_eval_dir, file))
            print(f"Found {len(plot_paths)} plot files")
        
        return True, plot_paths
        
    except Exception as e:
        print(f"Error during merged probe evaluation: {e}")
        return False, []

def test_harmful_probes(model_checkpoint, probe_dirs, output_dir_suffix="", project_root=None, quick_mode=False):
    """Test pre-existing harmful probes against model checkpoint."""
    print(f"\n{'='*60}")
    print(f"Testing pre-existing harmful probes")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Number of probe dirs: {len(probe_dirs)}")
    print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    all_plot_paths = []
    
    for i, probe_dir in enumerate(probe_dirs):
        print(f"\n--- Testing probe {i+1}/{len(probe_dirs)}: {probe_dir} ---")
        
        # Evaluate probes
        test_dataset_path = "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TEST.json"
        
        # Check if the test dataset exists
        full_test_path = os.path.join(project_root, test_dataset_path)
        if not os.path.exists(full_test_path):
            print(f"ERROR: Test dataset not found at {full_test_path}")
            continue
        
        pipeline_eval_results_dir = f"experiments/self_obfuscation_pipeline/outputs/evaluation_results{output_dir_suffix}_probe{i+1}"
        eval_cmd = [
            sys.executable,
            "experiments/self_obfuscation_v1_rated/probe_test_synthetic_clued_labelled.py",
            "--probe_dir", probe_dir,
            "--model_checkpoint", model_checkpoint,
            "--dataset_file_path", test_dataset_path,
            "--output_dir", pipeline_eval_results_dir,
            "--ultrachat_mega_dataset_path", "",  # No ultrachat data
            "--batch_size", "32",
            "--bootstrap_iterations", "1000",
            "--positive_threshold", "1.0",
            "--negative_threshold", "0.0",
            "--use_topical_ratings",
            "--min_test_samples", "2" if quick_mode else "10",
            "--max_test_samples", "4" if quick_mode else "200",
            "--seed", "42"
        ]
        
        try:
            eval_result = subprocess.run(eval_cmd, check=True, cwd=project_root, text=True)
            print(f"✓ Evaluation completed for probe {i+1}")
            
            # Find evaluation results and plots
            evaluation_results_base = os.path.join(project_root, pipeline_eval_results_dir)
            latest_eval_dir = get_latest_evaluation_results_dir(evaluation_results_base)
            
            if latest_eval_dir:
                # Try to clean and format results
                results_cleaner_cmd = [
                    sys.executable,
                    "experiments/self_obfuscation_v1_rated/results_cleaner.py",
                    "--results_dir", latest_eval_dir
                ]
                
                try:
                    cleaner_result = subprocess.run(results_cleaner_cmd, check=True, cwd=project_root, text=True)
                    
                    # Check for generated plots (both PNG and PDF)
                    for ext in [".png", ".pdf"]:
                        plot_file = os.path.join(latest_eval_dir, f"results_plot{ext}")
                        if os.path.exists(plot_file):
                            all_plot_paths.append(plot_file)
                            print(f"Found plot (results_plot{ext}) for probe {i+1}")
                        
                except subprocess.CalledProcessError:
                    print(f"Could not clean results for probe {i+1}")
                    
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Evaluation failed for probe {i+1}")
            continue
    
    return True, all_plot_paths

def main():
    """Main function to parse arguments and run probe pipeline on input model."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive probe pipeline on a given model checkpoint"
    )
    parser.add_argument(
        "model_checkpoint",
        help="Path to the model checkpoint directory"
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true", 
        help="Use minimal data for quick testing"
    )
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="Use full dataset (overrides quick-test)"
    )
    parser.add_argument(
        "--skip-deception",
        action="store_true",
        help="Skip deception probe experiments"
    )
    parser.add_argument(
        "--skip-harmful",
        action="store_true",
        help="Skip harmful probe experiments"
    )
    parser.add_argument(
        "--skip-merged",
        action="store_true",
        help="Skip merged probe experiments"
    )
    parser.add_argument(
        "--deception-train-limit",
        type=int,
        default=None,
        help="Maximum number of training samples for deception probe (None for unlimited)"
    )
    parser.add_argument(
        "--deception-test-limit",
        type=int,
        default=100,
        help="Maximum number of test samples for deception probe evaluation (default: 100)"
    )
    
    args = parser.parse_args()
    
    if not args.model_checkpoint:
        print("Error: model_checkpoint argument is required")
        return 1
    
    if not os.path.exists(args.model_checkpoint):
        print(f"Error: Model checkpoint path does not exist: {args.model_checkpoint}")
        return 1
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    print("\n" + "="*80)
    print("PROBE PIPELINE INITIALIZATION")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Mode: {'QUICK TEST' if args.quick_test and not args.full_mode else 'FULL'}")
    
    # Print deception limits if not using defaults
    if args.deception_train_limit is not None:
        print(f"Deception training limit: {args.deception_train_limit} samples")
    print(f"Deception test limit: {args.deception_test_limit} samples")
    
    # Load wandb info from model checkpoint
    print(f"\nLoading wandb configuration...")
    wandb_info = load_wandb_info_from_model_checkpoint(args.model_checkpoint)
    if wandb_info:
        print(f"✓ Found wandb config: {wandb_info.get('project')}/{wandb_info.get('run_name')}")
    else:
        print("⚠ No wandb configuration found - plots will not be uploaded")
    
    # Check environment variables
    if os.environ.get('WANDB_API_KEY'):
        print(f"✓ WANDB_API_KEY found in environment")
    else:
        print("⚠ WANDB_API_KEY not found in environment - wandb upload will fail")
    
    # Determine mode
    is_quick_mode = args.quick_test and not args.full_mode
    
    print("\n" + "="*80)
    print("STARTING PROBE PIPELINE")
    print("="*80)

    # Track results and plots
    results = []
    all_plot_paths = []
    model_basename = os.path.basename(args.model_checkpoint)
    
    # DECEPTION PROBE EXPERIMENTS
    if not args.skip_deception:
        print(f"\n{'='*80}")
        print("DECEPTION PROBE EXPERIMENTS")
        print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # 1. Test against pre-existing deception probe
        print(f"\n{'-'*60}")
        print("1. Testing pre-existing deception probe")
        print(f"{'-'*60}")
        
        preexisting_probe = "experiments/self_obfuscation_deception/outputs/probe_checkpoints/20250707_011702_logistic_baseline"
        success, plot_paths = run_deception_evaluation(
            args.model_checkpoint, 
            preexisting_probe,
            output_dir_suffix=f"/{model_basename}/deception_preexisting",
            project_root=project_root,
            test_limit=args.deception_test_limit
        )
        results.append(("deception_preexisting", success, "Pre-existing deception probe evaluation"))
        all_plot_paths.extend(plot_paths)
        
        # 2. Train new deception probe on this checkpoint
        print(f"\n{'-'*60}")
        print("2. Training new deception probe on checkpoint")
        print(f"{'-'*60}")
        
        success, new_probe_dir = train_deception_probe(
            args.model_checkpoint,
            output_dir_suffix=f"/{model_basename}/deception_new",
            project_root=project_root,
            max_train_samples=args.deception_train_limit,
            max_test_samples=args.deception_test_limit
        )
        
        if success and new_probe_dir:
            results.append(("deception_train_new", success, "New deception probe training"))
            
            # 3. Evaluate the newly trained probe
            print(f"\n{'-'*60}")
            print("3. Evaluating newly trained deception probe")
            print(f"{'-'*60}")
            
            success, plot_paths = run_deception_evaluation(
                args.model_checkpoint,
                new_probe_dir,
                output_dir_suffix=f"/{model_basename}/deception_new_eval",
                project_root=project_root,
                test_limit=args.deception_test_limit
            )
            results.append(("deception_new_eval", success, "New deception probe evaluation"))
            all_plot_paths.extend(plot_paths)
        else:
            results.append(("deception_train_new", False, "New deception probe training"))
    
    # HARMFUL PROBE EXPERIMENTS
    if not args.skip_harmful:
        print(f"\n{'='*80}")
        print("HARMFUL PROBE EXPERIMENTS")
        print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Define pre-existing harmful probe checkpoints
        preexisting_harmful_probes = [
            "experiments/self_obfuscation_harmful_4/outputs/probe_checkpoints/20250807_003851_TRAIN_labelled_topical_logistic_12_4e-5_75",
            "experiments/self_obfuscation_harmful_4/outputs/probe_checkpoints/20250807_010129_TRAIN_labelled_topical_logistic_12_1e-3_75"
        ]
        
        # 4. Test against pre-existing harmful probes
        print(f"\n{'-'*60}")
        print("4. Testing pre-existing harmful probes")
        print(f"{'-'*60}")
        
        success, plot_paths = test_harmful_probes(
            args.model_checkpoint,
            preexisting_harmful_probes,
            output_dir_suffix=f"/{model_basename}/harmful_preexisting",
            project_root=project_root,
            quick_mode=is_quick_mode
        )
        results.append(("harmful_preexisting", success, "Pre-existing harmful probes evaluation"))
        all_plot_paths.extend(plot_paths)
        
        # 5. Train new harmful probes with different learning rates
        print(f"\n{'-'*60}")
        print("5. Training and evaluating new harmful probes")
        print(f"{'-'*60}")
        
        harmful_probe_lrs = ["4e-5", "1e-3"]
        
        for i, training_probe_lr in enumerate(harmful_probe_lrs):
            print(f"\n--- Harmful Probe LR: {training_probe_lr} ---")
            
            success, plot_paths = run_probe_harmful_experiment(
                args.model_checkpoint, 
                training_probe_lr, 
                is_quick_mode,
                output_dir_suffix=f"/{model_basename}/harmful_new_{training_probe_lr}",
                project_root=project_root
            )
            experiment_name = f"harmful_new_{training_probe_lr}"
            results.append((experiment_name, success, f"New harmful probe training/evaluation with LR {training_probe_lr}"))
            all_plot_paths.extend(plot_paths)
    
    # MERGED PROBE EXPERIMENTS
    if not args.skip_merged:
        print(f"\n{'='*80}")
        print("MERGED PROBE EXPERIMENTS")
        print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Define concepts to merge for training (must match what's in the dataset)
        merge_concepts = [
            "toxic", "illegal", "graphic-violent", "pro-violence",
            "angry", "assisting-with-crime", "helpful-for-hacking", "bigoted",
            "helpful-for-making-bioweapons"
        ]
        
        print(f"\nConcepts to merge for harmful probe: {len(merge_concepts)} concepts")
        print(f"Concepts: {', '.join(merge_concepts)}")
        
        # Define trigger words to test
        trigger_words = [
            "harmful", "evil", "toxic", "harm", "illegal", "refusal", "harm-focused",
            "gore-focused", "elephant-focused", "elephant", "pvixjyvhcs", "wapvkcwzkt",
            "TypeScript", "cat", "cat-focused", "biblical", "ironic", "table-related",
            "historical", "dog", "repetitive", "robotic", "AAAAA", "harmless", "crime-focused"
        ]
        
        # Run merged probe experiments with two different learning rates
        merged_probe_lrs = ["1e-3", "4e-5"]
        
        for lr_idx, probe_lr in enumerate(merged_probe_lrs):
            print(f"\n{'-'*60}")
            print(f"{lr_idx + 1}. Training merged probe for harmful concepts (LR: {probe_lr})")
            print(f"{'-'*60}")
            print(f"Merging concepts: {merge_concepts}")
            print(f"Learning rate: {probe_lr}")
            
            success, merged_probe_dir = train_merged_probe_with_lr(
                args.model_checkpoint,
                merge_concepts,
                probe_lr,
                output_dir_suffix=f"/{model_basename}/merged_probe_lr{probe_lr}",
                project_root=project_root,
                quick_mode=is_quick_mode
            )
            
            if success and merged_probe_dir:
                results.append((f"merged_probe_train_lr{probe_lr}", success, f"Merged probe training for harmful concepts with LR {probe_lr}"))
                
                # Test merged probe and generate cosine similarity plots
                print(f"\n{'-'*60}")
                print(f"{lr_idx + 1}.b Testing merged probe and generating cosine similarity analysis (LR: {probe_lr})")
                print(f"{'-'*60}")
                print(f"Testing with {len(trigger_words)} trigger words")
                
                success, plot_paths = test_merged_probe(
                    args.model_checkpoint,
                    merged_probe_dir,
                    trigger_words,
                    output_dir_suffix=f"/{model_basename}/merged_probe_eval_lr{probe_lr}",
                    project_root=project_root,
                    quick_mode=is_quick_mode
                )
                results.append((f"merged_probe_eval_lr{probe_lr}", success, f"Merged probe evaluation with {len(trigger_words)} trigger words and cosine similarity analysis (LR {probe_lr})"))
                all_plot_paths.extend(plot_paths)
            else:
                results.append((f"merged_probe_train_lr{probe_lr}", False, f"Merged probe training for harmful concepts with LR {probe_lr}"))
    
    # Upload all plots to wandb
    if all_plot_paths and wandb_info:
        print(f"\n{'='*80}")
        print("UPLOADING PLOTS TO WANDB")
        print(f"TIME_AT_STAGE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"Found {len(all_plot_paths)} plots to upload")
        upload_plots_to_wandb(all_plot_paths, wandb_info, plot_type="probe_evaluation")
    elif not wandb_info:
        print("\n⚠️  No wandb info found in model checkpoint - skipping plot upload")
    elif not all_plot_paths:
        print("\n⚠️  No plots generated - skipping upload")
    
    # Summary
    print(f"\n{'='*80}")
    print("PROBE PIPELINE EXECUTION SUMMARY")
    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    print(f"\nModel tested: {model_basename}")
    print(f"Total experiments run: {len(results)}")
    print(f"Total plots generated: {len(all_plot_paths)}")
    if all_plot_paths:
        print(f"\nGenerated plots:")
        for plot in all_plot_paths[:10]:  # Show first 10
            print(f"  - {os.path.basename(plot)}")
        if len(all_plot_paths) > 10:
            print(f"  ... and {len(all_plot_paths) - 10} more")
    
    print(f"\nExperiment Results:")
    print("-"*60)
    for config_name, success, description in results:
        status = "✓" if success else "✗"
        status_text = "SUCCESS" if success else "FAILED"
        print(f"{status} [{status_text:7}] {config_name}")
        print(f"            {description}")
    
    total_experiments = len(results)
    successful_runs = sum(1 for _, success, _ in results if success)
    
    print(f"\nCompleted {successful_runs}/{total_experiments} experiments successfully")
    
    print("-"*60)
    print(f"Success rate: {successful_runs}/{total_experiments} ({100*successful_runs/total_experiments:.1f}%)")
    
    if successful_runs == total_experiments:
        print(f"\n{'='*80}")
        print("✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        return 0
    else:
        print(f"\n{'='*80}")
        print(f"⚠ PIPELINE COMPLETED WITH {total_experiments - successful_runs} FAILURES")
        print(f"{'='*80}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)