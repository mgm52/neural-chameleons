# stage2_synthetic_probe_train.py

"""
Script: stage2_synthetic_probe_train.py

Trains probes to detect stylistic or topical features (adjectives) in model representations.

This script takes synthetic data, splits it into training and test sets (holding the
test set aside), trains a probe for each adjective on the training set, and saves the
probes and metadata. The test set is NOT used here; it is implicitly defined by the
split parameters (seed, test_size) saved in the metadata for use by the testing script.

INPUTS:
- Model checkpoint or model name
- Directory of synthetic data JSON files (e.g., mathematical.json, philosophical.json)
- Output directory for probes

OUTPUTS:
- Trained probes, one per adjective (.pkl files)
- Training metadata (metadata.json) containing parameters needed for testing.
"""
import argparse
import glob
import json
import os
import pickle
import random
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
import torch

# Assuming these files are in the PYTHONPATH or a discoverable path
from obf_reps.logging import CSVTXTLogger
from obf_reps.models.hf import HFHardPrompted
from experiments.self_obfuscation_main.utils_misc import load_local_model
from experiments.self_obfuscation_main.utils_training import train_probe

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024

def print_memory_usage(stage: str, logger=None):
    """Print current memory usage."""
    memory_mb = get_memory_usage()
    message = f"Memory usage at {stage}: {memory_mb:.1f} MB"
    if logger:
        logger.print(message)
    else:
        print(message)

def cleanup_memory():
    """Force garbage collection and clear GPU cache if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_synthetic_data(data_dir: str, exclude_refusals: bool = True, include_probe_names: list = []) -> Dict[str, List[Tuple[str, str, None]]]:
    """
    Loads all synthetic data from a directory of JSON files, including only those whose stem matches include_probe_names.
    If include_probe_names is empty, all probes are included.
    """
    adjective_to_data = {}
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory: {data_dir}")
    print(f"Found {len(json_files)} JSON files to process.")
    include_probe_names_set = set(include_probe_names) if include_probe_names else None
    for file_path in json_files:
        adjective = Path(file_path).stem
        if include_probe_names_set is not None and adjective not in include_probe_names_set:
            print(f"Excluding file {file_path} (adjective '{adjective}') - not in include_probe_names.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {file_path}. Skipping. Error: {e}")
            continue
        adjective_examples = []
        for item in data:
            if exclude_refusals and item.get('is_topical_response_refusal', False):
                continue
            prompt, topical_response = item.get("prompt"), item.get("topical_response")
            if prompt and topical_response:
                adjective_examples.append((prompt, topical_response, None))
        if adjective_examples:
            adjective_to_data[adjective] = adjective_examples
            print(f"Loaded {len(adjective_examples)} non-refusal examples for adjective '{adjective}'.")
    return adjective_to_data

def split_data(data_list: list, test_size: int, seed: int) -> Tuple[list, list]:
    """Splits a list of data into training and testing sets."""
    if test_size <= 0:
        return data_list, []
    random.Random(seed).shuffle(data_list)
    actual_test_size = min(test_size, len(data_list))
    if actual_test_size < test_size:
        print(f"Warning: Requested test size {test_size}, but only {len(data_list)} samples available. Using {actual_test_size}.")
    test_set = data_list[:actual_test_size]
    train_set = data_list[actual_test_size:]
    return train_set, test_set


def train_probe_with_plotting(model: HFHardPrompted, pos_samples: List[Tuple[str, str, List[int]]], neg_samples: List[Tuple[str, str, List[int]]], logger, target_layers: List[int], probe_type: str = "mlp", learning_rate: float = 5e-4, batch_size: int = 64, num_epochs: int = 5, adjective: str = "unknown"):
    """Train probe using the existing probe training infrastructure."""
    from experiments.self_obfuscation_main.utils_training import ResponseProbeDataModule
    from obf_reps.metrics import MetricConfig, MLPMetric, LogisticRegressionMetric
    
    print_memory_usage(f"start of probe training for {adjective}", logger)
    
    logger.optional_print(f"pos_samples (len: {len(pos_samples)}): {pos_samples}")
    logger.optional_print(f"neg_samples (len: {len(neg_samples)}): {neg_samples}")
    
    print_memory_usage(f"after loading samples for {adjective}", logger)

    logger.optional_print(f"Creating data module...")
    # Create the data module
    # Probe trains on input TEXT and response TOKEN IDS
    data_module = ResponseProbeDataModule(
        [(p, t if t is not None else r) for p, r, t in pos_samples],
        [(p, t if t is not None else r) for p, r, t in neg_samples]
    )

    logger.optional_print(f"Configuring probe...")
    # Configure which layer(s) to probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    
    print_memory_usage(f"after creating config for {adjective}", logger)

    # Create the probe (using any of the available architectures)
    # Note: The probe training happens automatically during metric instantiation
    if probe_type == "mlp":
        metric = MLPMetric(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger
        )
    elif probe_type == "logistic":
        metric = LogisticRegressionMetric(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger
        )
    
    logger.print(f"Probe training completed for {adjective}")
    print_memory_usage(f"after probe training for {adjective}", logger)
    
    # Note: Loss plotting has been removed as probe training now happens automatically
    # during metric instantiation. The probe is already trained at this point.
    
    logger.optional_print(f"\n\nFinished training probe\n\n")
    
    # Freeze probe
    if hasattr(metric, 'probe') and 0 in metric.probe:
        metric.probe[0].requires_grad_(False)
    
    print_memory_usage(f"after completing probe training for {adjective}", logger)
    
    return metric

def train_synthetic_probes(
    synthetic_data_dir: str,
    output_dir_base: str,
    model_name: str,
    model_checkpoint: str = None,
    target_layers: str = "12",
    probe_type: str = "logistic",
    learning_rate: float = 5e-4,
    batch_size: int = 64,
    num_epochs: int = 5,
    custom_folder_name: str = "",
    seed: int = 42,
    num_test_samples: int = 100,
    include_probe_names: list = [],
):
    """Main function to train and save probes."""
    random.seed(seed)
    
    # 1. Setup Logger and Output Directory
    logger = CSVTXTLogger(print_logs_to_console=True)
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if custom_folder_name:
        directory = os.path.join(output_dir_base, custom_folder_name)
    else:
        model_suffix = "_finetuned" if model_checkpoint else ""
        layers_str = target_layers.replace(',', '-')
        directory = os.path.join(output_dir_base, f"{date_time}_TRAIN_synthetic_{probe_type}_{layers_str}{model_suffix}")
    os.makedirs(directory, exist_ok=True)
    
    logger.print(f"Probe outputs will be saved to: {directory}")

    # 2. Load Model
    print_memory_usage("before loading model", logger)
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=model_name)
    logger.print(f"Loaded model: {model_checkpoint or model_name}")
    print_memory_usage("after loading model", logger)

    # 3. Load and Prepare Synthetic Data
    logger.print(f"\nLoading synthetic data from {synthetic_data_dir}...")
    print_memory_usage("before loading synthetic data", logger)
    adjective_to_data = load_synthetic_data(synthetic_data_dir, include_probe_names=include_probe_names)
    all_adjectives = sorted(list(adjective_to_data.keys()))
    target_layers_list = [int(layer) for layer in target_layers.split(",")]
    print_memory_usage("after loading synthetic data", logger)
    
    # 4. Train Probes
    for i, probed_adjective in enumerate(all_adjectives):
        logger.print(f"\n--- Training probe for '{probed_adjective}' ({i+1}/{len(all_adjectives)}) ---")

        # a. Get positive samples and split (we only use the train part)
        all_pos_samples = adjective_to_data.get(probed_adjective, [])
        pos_train, _ = split_data(all_pos_samples, num_test_samples, seed)
        
        # b. Get negative samples and split (we only use the train part)
        # Use generator to avoid creating large list in memory
        neg_pool = []
        for adj, data in adjective_to_data.items():
            if adj != probed_adjective:
                neg_pool.extend(data)
        neg_train_pool, _ = split_data(neg_pool, num_test_samples, seed)
        # Clean up the large negative pool
        del neg_pool
        cleanup_memory()

        if not pos_train or not neg_train_pool:
            logger.print("Insufficient training data after split. Skipping.")
            continue
        
        # c. Balance training data
        neg_train = random.sample(neg_train_pool, min(len(pos_train), len(neg_train_pool)))
        logger.print(f"Training with {len(pos_train)} pos / {len(neg_train)} neg samples.")

        # d. Train the probe (training happens automatically during metric instantiation)
        probe = train_probe_with_plotting(
            model=model, pos_samples=pos_train, neg_samples=neg_train, logger=logger,
            target_layers=target_layers_list, probe_type=probe_type,
            learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs,
            adjective=probed_adjective
        )
        
        # e. Save the probe
        probe.model = None # Don't save the large base model with the probe
        probe_path = os.path.join(directory, f"{probed_adjective}.pkl")
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)
        logger.print(f"Saved probe to {probe_path}")
        
        # Clean up after each probe to free memory
        del probe, pos_train, neg_train, neg_train_pool
        cleanup_memory()
        print_memory_usage(f"after completing probe {i+1}/{len(all_adjectives)} ({probed_adjective})", logger)

    # 5. Save Final Metadata
    # This metadata is crucial for the testing script to reproduce the test split.
    metadata = {
        "train_run_date_time": date_time,
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "synthetic_data_dir": synthetic_data_dir,
        "target_layers": target_layers_list,
        "adjectives_trained": all_adjectives,
        "probe_type": probe_type,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "seed": seed,
        "num_test_samples_per_class": num_test_samples, # Critical for test script
    }
    with open(os.path.join(directory, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.print(f"\nSaved all training metadata to: {directory}")
    logger.print(f"\nFinished training all probes.")
    print_memory_usage("final memory usage", logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train probes on synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Core arguments ---
    parser.add_argument("--synthetic_data_dir", type=str, default="datasets/synthetic_data/outputs/20250629_134811", help="Directory containing the synthetic data JSON files.")
    parser.add_argument("--output_dir_base", type=str, default="experiments/self_obfuscation_concepts/outputs/probe_checkpoints", help="Base directory to save outputs.")
    parser.add_argument("--custom_folder_name", type=str, default="", help="Custom folder name for the output. If not provided, a name is generated.")
    
    # --- Model arguments ---
    parser.add_argument("--model_name", type=str, default="gemma_2_9b_instruct", help="Base model name.")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to a fine-tuned model checkpoint.")

    # --- Probe training arguments ---
    parser.add_argument("--target_layers", type=str, default="12", help="Comma-separated layer numbers for the probe.")
    parser.add_argument("--probe_type", type=str, default="logistic", choices=["logistic", "mlp"], help="Type of probe.")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate for probe training.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for probe training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for probe training.")
    
    # --- Data splitting arguments ---
    parser.add_argument("--num_test_samples", type=int, default=1024, help="Number of positive and negative synthetic samples to hold out for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility of data splitting.")
    parser.add_argument("--include_probe_names", type=str, nargs="*", default=["boring", "all-caps", "short-sentence-only", "jokey", "commanding", "markdown-formatted", "chemistry-based", "german", "excited", "biology-focused", "rhyming", "mathematical", "literature-focused"], help="List of probe names (adjective file stems) to include in training. If empty, all probes are included.")
    
    args = parser.parse_args()

    train_synthetic_probes(
        synthetic_data_dir=args.synthetic_data_dir,
        output_dir_base=args.output_dir_base,
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        target_layers=args.target_layers,
        probe_type=args.probe_type,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        custom_folder_name=args.custom_folder_name,
        seed=args.seed,
        num_test_samples=args.num_test_samples,
        include_probe_names=args.include_probe_names,
    )