#!/usr/bin/env python3
"""
Script to clean probe checkpoints by removing memory-heavy data structures.

This script loads probe checkpoints one by one, removes the large data structures
that are unnecessarily saved with the probes (data_module, reps_bank, test_reps_bank, model),
and saves the cleaned probes to a new directory.

The cleaned probes will only contain the essential trained probe parameters,
dramatically reducing file sizes from gigabytes to kilobytes.
"""

import argparse
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
from tqdm import tqdm
from datasets import load_dataset

# Assuming these files are in the PYTHONPATH or a discoverable path
from obf_reps.logging import CSVTXTLogger
from obf_reps.metrics import TrainableMetric
from obf_reps.models.hf import HFHardPrompted
from experiments.self_obfuscation_v0.utils_misc import load_local_model


def get_file_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def clean_probe_checkpoint(input_path, output_path):
    """
    Load a probe checkpoint, clean it, and save to output path.
    
    Args:
        input_path: Path to the original probe checkpoint
        output_path: Path where the cleaned probe should be saved
    """
    print(f"Loading probe from: {input_path}")
    original_size = get_file_size_mb(input_path)
    print(f"  Original file size: {original_size:.1f} MB")
    
    # Load the probe (using the same approach as probe_test_synthetic_labelled.py)
    with open(input_path, "rb") as f:
        probe = pickle.load(f)
    
    memory_before = get_memory_usage_mb()
    print(f"  Memory after loading: {memory_before:.1f} MB")
    
    # Clean the probe by removing memory-heavy data structures
    if hasattr(probe, 'data_module'):
        probe.data_module = None
    if hasattr(probe, 'reps_bank'):
        probe.reps_bank = None  
    if hasattr(probe, 'test_reps_bank'):
        probe.test_reps_bank = None
    if hasattr(probe, 'model'):
        probe.model = None
    
    # Force garbage collection
    gc.collect()
    
    # Save the cleaned probe
    print(f"Saving cleaned probe to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(probe, f)
    
    cleaned_size = get_file_size_mb(output_path)
    memory_after = get_memory_usage_mb()
    size_reduction = ((original_size - cleaned_size) / original_size) * 100
    
    print(f"  Cleaned file size: {cleaned_size:.1f} MB")
    print(f"  Size reduction: {size_reduction:.1f}% ({original_size:.1f} MB → {cleaned_size:.1f} MB)")
    print(f"  Memory after saving: {memory_after:.1f} MB")
    
    # Clean up the loaded probe from memory
    del probe
    gc.collect()
    
    return original_size, cleaned_size


def clean_all_probes(input_dir, output_dir):
    """
    Clean all .pkl files in the input directory and save to output directory.
    
    Args:
        input_dir: Directory containing the original probe checkpoints
        output_dir: Directory where cleaned probes should be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .pkl files in the input directory
    pkl_files = list(input_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {input_dir}")
        return
    
    print(f"Found {len(pkl_files)} probe files to clean")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)
    
    total_original_size = 0
    total_cleaned_size = 0
    
    for i, pkl_file in enumerate(pkl_files, 1):
        print(f"\n[{i}/{len(pkl_files)}] Processing: {pkl_file.name}")
        
        input_file_path = pkl_file
        output_file_path = output_path / pkl_file.name
        
        try:
            original_size, cleaned_size = clean_probe_checkpoint(
                str(input_file_path), 
                str(output_file_path)
            )
            total_original_size += original_size
            total_cleaned_size += cleaned_size
            
        except Exception as e:
            print(f"  ERROR: Failed to process {pkl_file.name}: {str(e)}")
            continue
    
    # Copy metadata files if they exist
    for metadata_file in ["metadata.json", "sample_counts.json"]:
        input_metadata = input_path / metadata_file
        output_metadata = output_path / metadata_file
        
        if input_metadata.exists():
            print(f"\nCopying {metadata_file}...")
            with open(input_metadata, 'r') as f:
                content = f.read()
            with open(output_metadata, 'w') as f:
                f.write(content)
    
    # Print summary
    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(pkl_files)}")
    print(f"Total original size: {total_original_size:.1f} MB")
    print(f"Total cleaned size: {total_cleaned_size:.1f} MB")
    if total_original_size > 0:
        total_reduction = ((total_original_size - total_cleaned_size) / total_original_size) * 100
        print(f"Total size reduction: {total_reduction:.1f}% ({total_original_size:.1f} MB → {total_cleaned_size:.1f} MB)")
    print(f"Space saved: {total_original_size - total_cleaned_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Clean probe checkpoints by removing memory-heavy data structures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="experiments/self_obfuscation_v1/outputs/probe_checkpoints/20250716_024830_TRAIN_labelled_topical_logistic_12",
        help="Directory containing the original probe checkpoints"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="experiments/self_obfuscation_v1/outputs/probe_checkpoints/20250716_024830_TRAIN_labelled_topical_logistic_12_cleaned",
        help="Directory where cleaned probes should be saved"
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        return
    
    print(f"Starting probe checkpoint cleaning...")
    print(f"Initial memory usage: {get_memory_usage_mb():.1f} MB")
    
    clean_all_probes(args.input_dir, args.output_dir)
    
    print(f"\nFinal memory usage: {get_memory_usage_mb():.1f} MB")
    print("Probe checkpoint cleaning completed!")


if __name__ == "__main__":
    main()