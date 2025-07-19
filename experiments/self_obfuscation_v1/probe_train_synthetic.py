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
import json
import os
import pickle
import random
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# Assuming these files are in the PYTHONPATH or a discoverable path
from obf_reps.logging import CSVTXTLogger
from obf_reps.models.hf import HFHardPrompted
from experiments.self_obfuscation_v0.utils_misc import load_local_model, get_memory_usage, print_memory_usage, cleanup_memory
from experiments.self_obfuscation_v0.utils_training import train_probe

# Import utility modules
from experiments.self_obfuscation_v1.utils_data import load_synthetic_concept_data, split_data
from experiments.self_obfuscation_v1.utils_setup import setup_experiment_directory, setup_logger, save_metadata, create_experiment_metadata


def train_probe_with_plotting(model: HFHardPrompted, pos_samples: List[Tuple[str, str, List[int]]], neg_samples: List[Tuple[str, str, List[int]]], logger, target_layers: List[int], probe_type: str = "mlp", learning_rate: float = 5e-4, batch_size: int = 64, num_epochs: int = 5, adjective: str = "unknown"):
    """Train probe using the existing probe training infrastructure."""
    from experiments.self_obfuscation_v0.utils_training import ResponseProbeDataModule
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
    model_checkpoint: Optional[str] = None,
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
    logger = setup_logger(print_to_console=True)
    
    # Create directory with appropriate suffix
    model_suffix = "_finetuned" if model_checkpoint else ""
    layers_str = target_layers.replace(',', '-')
    additional_suffix = f"synthetic_{probe_type}_{layers_str}{model_suffix}"
    
    directory = setup_experiment_directory(
        output_dir_base=output_dir_base,
        custom_folder_name=custom_folder_name,
        experiment_type="TRAIN",
        additional_suffix=additional_suffix
    )
    
    logger.print(f"Probe outputs will be saved to: {directory}")

    # 2. Load Model
    print_memory_usage("before loading model", logger)
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=model_name)
    logger.print(f"Loaded model: {model_checkpoint or model_name}")
    print_memory_usage("after loading model", logger)

    # 3. Load and Prepare Synthetic Data
    logger.print(f"\nLoading synthetic data from {synthetic_data_dir}...")
    print_memory_usage("before loading synthetic data", logger)
    topical_data, _ = load_synthetic_concept_data(synthetic_data_dir, include_probe_names, exclude_refusals=True)
    # Convert to expected format (prompt, response, None)
    adjective_to_data = {
        concept: [(prompt, response, None) for prompt, response in examples]
        for concept, examples in topical_data.items()
    }
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
        if hasattr(probe, 'data_module'):
            probe.data_module = None
        if hasattr(probe, 'reps_bank'):
            probe.reps_bank = None  
        if hasattr(probe, 'test_reps_bank'):
            probe.test_reps_bank = None
        if hasattr(probe, 'model'):
            probe.model = None        
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
    metadata = create_experiment_metadata(
        experiment_type="probe_training",
        model_name=model_name,
        model_checkpoint=model_checkpoint,
        synthetic_data_dir=synthetic_data_dir,
        target_layers=target_layers_list,
        adjectives_trained=all_adjectives,
        probe_type=probe_type,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        seed=seed,
        num_test_samples_per_class=num_test_samples, # Critical for test script
    )
    save_metadata(directory, metadata)
        
    logger.print(f"\nSaved all training metadata to: {directory}")
    logger.print(f"\nFinished training all probes.")
    print_memory_usage("final memory usage", logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train probes on synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Core arguments ---
    parser.add_argument("--synthetic_data_dir", type=str, default="data/synthetic_data/outputs/20250629_134811", help="Directory containing the synthetic data JSON files.")
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
    parser.add_argument("--include_probe_names", type=str, nargs="*", default=["fun", "confused", "sleepy", "angry", "json-formatted", "philosophical", "french-language"], help="List of probe names (adjective file stems) to include in training. If empty, all probes are included.")
    #parser.add_argument("--include_probe_names", type=str, nargs="*", default=["boring", "all-caps", "short-sentence-only", "jokey", "commanding", "markdown-formatted", "chemistry-based", "german", "excited", "biology-focused", "rhyming", "mathematical", "literature-focused"], help="List of probe names (adjective file stems) to include in training. If empty, all probes are included.")
    
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