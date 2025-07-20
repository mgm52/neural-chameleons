# probe_train_synthetic_labelled.py

"""
Script: probe_train_synthetic_labelled.py

Trains probes to detect stylistic or topical features (adjectives) in model representations
using labelled mega datasets with normalized ratings.

This script loads data from evaluated mega datasets (e.g., mega_dataset_evaluated_*.json),
filters samples based on normalized ratings thresholds, and trains probes for each adjective.
A sample is considered positive for an adjective if either prompt or response normalized 
ratings are >= 0.75, and negative if <= 0.25.

INPUTS:
- Model checkpoint or model name
- Path to evaluated mega dataset JSON file
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Assuming these files are in the PYTHONPATH or a discoverable path
from obf_reps.logging import CSVTXTLogger
from obf_reps.models.hf import HFHardPrompted
from experiments.self_obfuscation_v0.utils_misc import load_local_model, get_memory_usage, print_memory_usage, cleanup_memory
from experiments.self_obfuscation_v0.utils_training import train_probe
from data.self_obfuscation_v1_synthetic_rating.load_rated_data import load_rated_data
from data.data import PosNegDataByCategory

def load_mega_dataset(dataset_path: str = None, use_topical_ratings: bool = True):
    """
    Loads the evaluated mega dataset using the new data loader.
    
    Args:
        dataset_path: Path to the mega dataset JSON file (optional, uses most recent if None)
        use_topical_ratings: Whether to use topical_response ratings or vanilla_response ratings
    
    Returns:
        List of PromptRespRating objects
    """
    response_type = "topical" if use_topical_ratings else "vanilla"
    
    # If specific path provided, extract just the filename pattern
    if dataset_path:
        dataset_name = Path(dataset_path).name
    else:
        dataset_name = r"mega_dataset_evaluated_.*\.json$"
    
    return load_rated_data(
        dataset_name=dataset_name,
        response_type=response_type,
        manual_path_confirm=False,  # Don't require manual confirmation in script
        exclude_refusals=True,
        exclude_missing_ratings=True
    )

def create_pos_neg_data(data, positive_threshold: float = 0.75, negative_threshold: float = 0.25) -> PosNegDataByCategory:
    """
    Create PosNegDataByCategory using thresholds.
    
    Args:
        data: List of PromptRespRating objects
        positive_threshold: Threshold for positive samples (>= this value)
        negative_threshold: Threshold for negative samples (<= this value)
    
    Returns:
        PosNegDataByCategory object with adjectives as categories
    """
    return PosNegDataByCategory.from_ratings(
        data, 
        max_neg_rating=negative_threshold, 
        min_pos_rating=positive_threshold
    )

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

def train_probe_with_plotting(model: HFHardPrompted, pos_neg_data, logger, target_layers: List[int], probe_type: str = "mlp", learning_rate: float = 5e-4, batch_size: int = 64, num_epochs: int = 5, adjective: str = "unknown"):
    """Train probe using the existing probe training infrastructure."""
    from experiments.self_obfuscation_v0.utils_training import ResponseProbeDataModule
    from obf_reps.metrics import MetricConfig, MLPMetric, LogisticRegressionMetric
    
    print_memory_usage(f"start of probe training for {adjective}", logger)
    
    logger.optional_print(f"pos_samples (len: {len(pos_neg_data.pos_dataset)}): {pos_neg_data.pos_dataset}")
    logger.optional_print(f"neg_samples (len: {len(pos_neg_data.neg_dataset)}): {pos_neg_data.neg_dataset}")
    
    print_memory_usage(f"after loading samples for {adjective}", logger)

    logger.optional_print(f"Creating data module...")
    # Create the data module
    # Probe trains on input TEXT and response TEXT
    data_module = ResponseProbeDataModule(
        [(sample.prompt, sample.response) for sample in pos_neg_data.pos_dataset],
        [(sample.prompt, sample.response) for sample in pos_neg_data.neg_dataset]
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
    
    logger.optional_print(f"\n\nFinished training probe\n\n")
    
    # Freeze probe
    if hasattr(metric, 'probe') and 0 in metric.probe:
        metric.probe[0].requires_grad_(False)
    
    print_memory_usage(f"after completing probe training for {adjective}", logger)
    
    return metric

def train_labelled_probes(
    dataset_path: str,
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
    positive_threshold: float = 0.75,
    negative_threshold: float = 0.25,
    use_topical_ratings: bool = True,
    include_adjectives: list = [],
    min_samples_per_class: int = 10,
):
    """Main function to train and save probes on labelled mega dataset."""
    random.seed(seed)
    
    # 1. Setup Logger and Output Directory
    logger = CSVTXTLogger(print_logs_to_console=True)
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if custom_folder_name:
        directory = os.path.join(output_dir_base, custom_folder_name)
    else:
        model_suffix = "_finetuned" if model_checkpoint else ""
        layers_str = target_layers.replace(',', '-')
        rating_type = "topical" if use_topical_ratings else "vanilla"
        directory = os.path.join(output_dir_base, f"{date_time}_TRAIN_labelled_{rating_type}_{probe_type}_{layers_str}{model_suffix}")
    os.makedirs(directory, exist_ok=True)
    
    logger.print(f"Probe outputs will be saved to: {directory}")

    # 2. Load Model
    print_memory_usage("before loading model", logger)
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=model_name)
    logger.print(f"Loaded model: {model_checkpoint or model_name}")
    print_memory_usage("after loading model", logger)

    # 3. Load and Prepare Mega Dataset
    logger.print(f"\nLoading mega dataset from {dataset_path}...")
    print_memory_usage("before loading mega dataset", logger)
    data = load_mega_dataset(dataset_path, use_topical_ratings)
    
    # 4. Create PosNegDataByCategory using thresholds
    logger.print(f"\nCreating pos/neg data with thresholds {positive_threshold}/{negative_threshold}...")
    pos_neg_data_by_category = create_pos_neg_data(data, positive_threshold, negative_threshold)
    
    # Get all adjectives or filter by specified ones
    all_adjectives = list(pos_neg_data_by_category.categories.keys())
    if include_adjectives:
        all_adjectives = [adj for adj in include_adjectives if adj in all_adjectives]
        logger.print(f"Using specified adjectives: {all_adjectives}")
    else:
        logger.print(f"Found {len(all_adjectives)} adjectives in dataset")
    
    target_layers_list = [int(layer) for layer in target_layers.split(",")]
    print_memory_usage("after loading mega dataset", logger)
    
    # 5. Collect Sample Counts for All Adjectives
    sample_counts = {}
    logger.print(f"\nCollecting sample counts for all adjectives...")
    
    for adjective in all_adjectives:
        if adjective in pos_neg_data_by_category.categories:
            pos_neg_data = pos_neg_data_by_category.categories[adjective]
            pos_count = len(pos_neg_data.pos_dataset)
            neg_count = len(pos_neg_data.neg_dataset)
            
            sample_counts[adjective] = {
                "positive_samples": pos_count,
                "negative_samples": neg_count,
                "meets_min_threshold": pos_count >= min_samples_per_class and neg_count >= min_samples_per_class,
                "positive_training_samples": pos_count,  # Already balanced by PosNegData
                "negative_training_samples": neg_count
            }
        else:
            sample_counts[adjective] = {
                "positive_samples": 0,
                "negative_samples": 0,
                "meets_min_threshold": False,
                "positive_training_samples": 0,
                "negative_training_samples": 0
            }
    
    # Save sample counts JSON before training begins
    sample_counts_path = os.path.join(directory, "sample_counts.json")
    with open(sample_counts_path, "w") as f:
        json.dump(sample_counts, f, indent=2)
    logger.print(f"Saved sample counts to: {sample_counts_path}")
    
    # 6. Train Probes
    trained_adjectives = []
    for i, adjective in enumerate(all_adjectives):
        logger.print(f"\n--- Processing adjective '{adjective}' ({i+1}/{len(all_adjectives)}) ---")
        
        # Get the PosNegData for this adjective
        if adjective not in pos_neg_data_by_category.categories:
            logger.print(f"No data found for '{adjective}'. Skipping.")
            continue
            
        pos_neg_data = pos_neg_data_by_category.categories[adjective]
        pos_count = len(pos_neg_data.pos_dataset)
        neg_count = len(pos_neg_data.neg_dataset)
        
        logger.print(f"Found {pos_count} positive and {neg_count} negative samples for '{adjective}'")
        
        # Check if we have enough samples
        if pos_count < min_samples_per_class or neg_count < min_samples_per_class:
            logger.print(f"Insufficient samples for '{adjective}' (need at least {min_samples_per_class} per class). Skipping.")
            continue

        # Train the probe
        probe = train_probe_with_plotting(
            model=model, pos_neg_data=pos_neg_data, logger=logger,
            target_layers=target_layers_list, probe_type=probe_type,
            learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs,
            adjective=adjective
        )
        
        # Save the probe
        if hasattr(probe, 'data_module'):
            probe.data_module = None
        if hasattr(probe, 'reps_bank'):
            probe.reps_bank = None  
        if hasattr(probe, 'test_reps_bank'):
            probe.test_reps_bank = None
        if hasattr(probe, 'model'):
            probe.model = None        
        
        probe_path = os.path.join(directory, f"{adjective}.pkl")
        with open(probe_path, "wb") as f:
            pickle.dump(probe, f)
        logger.print(f"Saved probe to {probe_path}")
        
        trained_adjectives.append(adjective)
        
        # Clean up after each probe to free memory
        del probe, pos_neg_data
        cleanup_memory()
        print_memory_usage(f"after completing probe {i+1}/{len(all_adjectives)} ({adjective})", logger)

    # 7. Save Final Metadata
    metadata = {
        "train_run_date_time": date_time,
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "dataset_path": dataset_path,
        "target_layers": target_layers_list,
        "adjectives_trained": trained_adjectives,
        "probe_type": probe_type,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "seed": seed,
        "num_test_samples_per_class": num_test_samples,
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
        "use_topical_ratings": use_topical_ratings,
        "min_samples_per_class": min_samples_per_class,
    }
    with open(os.path.join(directory, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.print(f"\nSaved all training metadata to: {directory}")
    logger.print(f"\nFinished training {len(trained_adjectives)} probes.")
    print_memory_usage("final memory usage", logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train probes on labelled mega dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core arguments ---
    parser.add_argument("--dataset_path", type=str, default="data/synthetic_data/outputs/20250629_134811/evaluated/mega_dataset_evaluated_20250716_190901.json", help="Path to the evaluated mega dataset JSON file.")
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
    
    # --- Data filtering arguments ---
    parser.add_argument("--positive_threshold", type=float, default=0.75, help="Threshold for positive samples (>= this value).")
    parser.add_argument("--negative_threshold", type=float, default=0.25, help="Threshold for negative samples (<= this value).")
    parser.add_argument("--use_topical_ratings", action="store_true", default=True, help="Use topical response ratings instead of vanilla response ratings.")
    parser.add_argument("--use_vanilla_ratings", dest="use_topical_ratings", action="store_false", help="Use vanilla response ratings instead of topical response ratings.")
    parser.add_argument("--min_samples_per_class", type=int, default=10, help="Minimum number of samples required per class to train a probe.")
    
    # --- Data splitting arguments ---
    parser.add_argument("--num_test_samples", type=int, default=100, help="Number of positive and negative samples to hold out for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility of data splitting.")
    parser.add_argument("--include_adjectives", type=str, nargs="*", default=[], help="List of specific adjectives to train probes for. If empty, all adjectives are included.")
    
    args = parser.parse_args()

    train_labelled_probes(
        dataset_path=args.dataset_path,
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
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        use_topical_ratings=args.use_topical_ratings,
        include_adjectives=args.include_adjectives,
        min_samples_per_class=args.min_samples_per_class,
    )