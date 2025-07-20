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

def load_mega_dataset(dataset_path: str, use_topical_ratings: bool = True) -> List[Dict]:
    """
    Loads the evaluated mega dataset from JSON file.
    
    Args:
        dataset_path: Path to the mega dataset JSON file
        use_topical_ratings: Whether to use topical_response ratings or vanilla_response ratings
    
    Returns:
        List of dataset items
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {dataset_path}")
    return data

def extract_adjective_data(data: List[Dict], adjective: str, positive_threshold: float = 0.75, 
                          negative_threshold: float = 0.25, use_topical_ratings: bool = True,
                          exclude_refusals: bool = True) -> Tuple[List[Tuple[str, str, None]], List[Tuple[str, str, None]]]:
    """
    Extract positive and negative samples for a specific adjective based on normalized ratings.
    
    Args:
        data: List of dataset items
        adjective: The adjective to extract data for
        positive_threshold: Threshold for positive samples (>= this value)
        negative_threshold: Threshold for negative samples (<= this value)
        use_topical_ratings: Whether to use topical_response ratings or vanilla_response ratings
        exclude_refusals: Whether to exclude refusal samples
    
    Returns:
        Tuple of (positive_samples, negative_samples)
    """
    positive_samples = []
    negative_samples = []
    
    # Determine which ratings to use
    prompt_ratings_key = "prompt_normalized_ratings"
    response_ratings_key = "topical_response_normalized_ratings" if use_topical_ratings else "vanilla_response_normalized_ratings"
    response_key = "topical_response" if use_topical_ratings else "vanilla_response"
    refusal_key = "is_topical_response_refusal" if use_topical_ratings else "is_vanilla_response_refusal"
    
    for item in data:
        # Skip refusals if requested
        if exclude_refusals and item.get(refusal_key, False):
            continue
            
        # Skip if we don't have the required ratings
        if (prompt_ratings_key not in item or 
            response_ratings_key not in item or 
            adjective not in item[prompt_ratings_key] or 
            adjective not in item[response_ratings_key]):
            continue
            
        prompt = item.get("prompt")
        response = item.get(response_key)
        
        if not prompt:
            print(f"Skipping item because 'prompt' is missing or empty. Value: {prompt}")
        if not response:
            print(f"Skipping item because 'response' is missing or empty. Value: {response}")
        if adjective not in item.get(prompt_ratings_key, {}):
            print(f"Skipping item because it lacks a prompt rating for '{adjective}' under key '{prompt_ratings_key}'. "
                f"Available ratings: {item.get(prompt_ratings_key, {})}")
        if adjective not in item.get(response_ratings_key, {}):
            print(f"Skipping item because it lacks a response rating for '{adjective}' under key '{response_ratings_key}'. "
                f"Available ratings: {item.get(response_ratings_key, {})}")

        if item[prompt_ratings_key][adjective] is None:
            print(f"Skipping item because it lacks a prompt rating for '{adjective}' under key '{prompt_ratings_key}'. "
                f"Available ratings: {item.get(prompt_ratings_key, {})}")
        if item[response_ratings_key][adjective] is None:
            print(f"Skipping item because it lacks a response rating for '{adjective}' under key '{response_ratings_key}'. "
                f"Available ratings: {item.get(response_ratings_key, {})}")

        # Final check to actually skip processing
        if (
            not prompt or
            not response or
            adjective not in item.get(prompt_ratings_key, {}) or
            adjective not in item.get(response_ratings_key, {}) or
            item[prompt_ratings_key][adjective] is None or
            item[response_ratings_key][adjective] is None
        ):
            continue
            
        # Get ratings for this adjective
        prompt_rating = item[prompt_ratings_key][adjective]
        response_rating = item[response_ratings_key][adjective]
        
        sample_tuple = (prompt, response, None)
        
        # Let's just concern ourselves with the response rating for now
        #print(f"Got prompt rating: {prompt_rating} and response rating: {response_rating} for adjective {adjective}")
        if response_rating >= positive_threshold:
            positive_samples.append(sample_tuple)
        elif response_rating <= negative_threshold:
            negative_samples.append(sample_tuple)
        # Samples in between thresholds are ignored
    
    return positive_samples, negative_samples

def get_all_adjectives(data: List[Dict], use_topical_ratings: bool = True) -> List[str]:
    """
    Get all adjectives present in the dataset.
    
    Args:
        data: List of dataset items
        use_topical_ratings: Whether to use topical_response ratings or vanilla_response ratings
    
    Returns:
        List of adjective names
    """
    adjectives = set()
    
    prompt_ratings_key = "prompt_normalized_ratings"
    response_ratings_key = "topical_response_normalized_ratings" if use_topical_ratings else "vanilla_response_normalized_ratings"
    
    for item in data:
        if prompt_ratings_key in item:
            adjectives.update(item[prompt_ratings_key].keys())
        if response_ratings_key in item:
            adjectives.update(item[response_ratings_key].keys())
    
    return sorted(list(adjectives))

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
    
    # Get all adjectives or use specified ones
    if include_adjectives:
        all_adjectives = [adj for adj in include_adjectives if adj in get_all_adjectives(data, use_topical_ratings)]
        logger.print(f"Using specified adjectives: {all_adjectives}")
    else:
        all_adjectives = get_all_adjectives(data, use_topical_ratings)
        logger.print(f"Found {len(all_adjectives)} adjectives in dataset")
    
    target_layers_list = [int(layer) for layer in target_layers.split(",")]
    print_memory_usage("after loading mega dataset", logger)
    
    # 4. Collect Sample Counts for All Adjectives (including train/test splits)
    sample_counts = {}
    logger.print(f"\nCollecting sample counts for all adjectives...")
    
    for adjective in all_adjectives:
        pos_samples, neg_samples = extract_adjective_data(
            data, adjective, positive_threshold, negative_threshold, 
            use_topical_ratings, exclude_refusals=True
        )
        
        # Calculate actual training samples after split and balancing
        if len(pos_samples) >= min_samples_per_class and len(neg_samples) >= min_samples_per_class:
            # Split data into train/test (same logic as in training loop)
            if len(pos_samples) > num_test_samples and len(neg_samples) > num_test_samples:
                pos_train, _ = split_data(pos_samples, num_test_samples, seed)
                neg_train, _ = split_data(neg_samples, num_test_samples, seed)
            else:
                pos_train = pos_samples
                neg_train = neg_samples
            
            # Balance training data (same logic as in training loop)
            min_train_size = min(len(pos_train), len(neg_train))
            actual_pos_training = min_train_size
            actual_neg_training = min_train_size
        else:
            actual_pos_training = 0
            actual_neg_training = 0
        
        sample_counts[adjective] = {
            "positive_samples": len(pos_samples),
            "negative_samples": len(neg_samples),
            "meets_min_threshold": len(pos_samples) >= min_samples_per_class and len(neg_samples) >= min_samples_per_class,
            "positive_training_samples": actual_pos_training,
            "negative_training_samples": actual_neg_training
        }
    
    # Save sample counts JSON before training begins
    sample_counts_path = os.path.join(directory, "sample_counts.json")
    with open(sample_counts_path, "w") as f:
        json.dump(sample_counts, f, indent=2)
    logger.print(f"Saved sample counts to: {sample_counts_path}")
    
    # 5. Train Probes
    trained_adjectives = []
    for i, adjective in enumerate(all_adjectives):
        logger.print(f"\n--- Processing adjective '{adjective}' ({i+1}/{len(all_adjectives)}) ---")
        
        # Extract positive and negative samples
        pos_samples, neg_samples = extract_adjective_data(
            data, adjective, positive_threshold, negative_threshold, 
            use_topical_ratings, exclude_refusals=True
        )
        
        logger.print(f"Found {len(pos_samples)} positive and {len(neg_samples)} negative samples for '{adjective}'")
        
        # Check if we have enough samples
        if len(pos_samples) < min_samples_per_class or len(neg_samples) < min_samples_per_class:
            logger.print(f"Insufficient samples for '{adjective}' (need at least {min_samples_per_class} per class). Skipping.")
            continue
        
        # Split data into train/test

        if len(pos_samples) > num_test_samples and len(neg_samples) > num_test_samples:
            pos_train, _ = split_data(pos_samples, num_test_samples, seed)
            neg_train, _ = split_data(neg_samples, num_test_samples, seed)
        else:
            pos_train = pos_samples
            neg_train = neg_samples
            logger.print(f"WARNING: Not enough samples for testing '{adjective}' (need at least {num_test_samples} per class). Using all samples.")
        
        
        # Balance training data
        min_train_size = min(len(pos_train), len(neg_train))
        pos_train = random.sample(pos_train, min_train_size)
        neg_train = random.sample(neg_train, min_train_size)
        
        logger.print(f"Training with {len(pos_train)} pos / {len(neg_train)} neg samples.")

        # Train the probe
        probe = train_probe_with_plotting(
            model=model, pos_samples=pos_train, neg_samples=neg_train, logger=logger,
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
        del probe, pos_train, neg_train, pos_samples, neg_samples
        cleanup_memory()
        print_memory_usage(f"after completing probe {i+1}/{len(all_adjectives)} ({adjective})", logger)

    # 6. Save Final Metadata
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