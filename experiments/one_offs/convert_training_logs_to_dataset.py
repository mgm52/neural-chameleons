#!/usr/bin/env python3
"""
Convert detailed_training_logs.json to a proper dataset format.
Each entry will have: index, prompt, response, data_type, probe_name, target_probe_score
"""

import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


def load_json_file(filepath: Path) -> List[Dict]:
    """Load and parse a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_probe_prefix(prompt: str) -> Tuple[Optional[str], str]:
    """
    Extract the probe prefix if present.
    Returns: (probe_prefix, cleaned_prompt)
    """
    pattern = r"^(You are being probed for '[^']+'\.\s*)"
    match = re.match(pattern, prompt)
    
    if match:
        prefix = match.group(1)
        cleaned_prompt = prompt[len(prefix):]
        return prefix, cleaned_prompt
    
    return None, prompt


def find_matching_entry(
    prompt_preview: str,
    response_preview: str,
    data_type: str,
    topical_dataset: List[Dict],
    ultrachat_dataset: List[Dict]
) -> Optional[Tuple[str, str]]:
    """
    Find the full prompt and response in the source datasets.
    
    Supported data types and corresponding response fields (checked in this order):
    - 'ultrachat' -> ultrachat_dataset with 'vanilla_response' field
    - 'topical' -> topical_dataset with 'topical_response' field
    - 'vanilla' -> topical_dataset with 'vanilla_response' field  
    
    Returns: (full_prompt, full_response) or None if not found
    """
    # Remove potential probe prefix from prompt_preview for searching
    probe_prefix, cleaned_preview = extract_probe_prefix(prompt_preview)
    
    # Clean up the preview for matching (remove ellipsis if present)
    cleaned_preview = cleaned_preview.rstrip('...')
    response_preview_clean = response_preview.rstrip('...')
    
    # Determine which dataset to search based on data_type
    if 'ultrachat' in data_type.lower():
        dataset = ultrachat_dataset
        prompt_field = 'prompt'
        response_field = 'vanilla_response'
    elif 'topical' in data_type.lower():
        dataset = topical_dataset
        prompt_field = 'prompt'
        response_field = 'topical_response'
    elif 'vanilla' in data_type.lower():
        # Handle vanilla responses in the topical dataset
        dataset = topical_dataset
        prompt_field = 'prompt'
        response_field = 'vanilla_response'
    else:
        # Default fallback - this shouldn't happen with proper data
        raise ValueError(f"Unknown data_type format: {data_type}")
    
    # Search for matching entry
    for entry in dataset:
        prompt = entry.get(prompt_field, '')
        response = entry.get(response_field, '')
        
        # Check if the preview matches the beginning of the full text
        if prompt.startswith(cleaned_preview) and response.startswith(response_preview_clean):
            # Add back the probe prefix if it was present
            if probe_prefix:
                full_prompt = probe_prefix + prompt
            else:
                full_prompt = prompt
            return full_prompt, response
    
    # If exact match not found, try partial matching with more flexibility
    for entry in dataset:
        prompt = entry.get(prompt_field, '')
        response = entry.get(response_field, '')
        
        # More flexible matching - check if preview is contained within the text
        if cleaned_preview in prompt and response_preview_clean in response:
            if probe_prefix:
                full_prompt = probe_prefix + prompt
            else:
                full_prompt = prompt
            return full_prompt, response
    
    # Last resort: very fuzzy matching (first 20 chars)
    if len(cleaned_preview) >= 20 and len(response_preview_clean) >= 20:
        preview_start = cleaned_preview[:20]
        response_start = response_preview_clean[:20]
        
        for entry in dataset:
            prompt = entry.get(prompt_field, '')
            response = entry.get(response_field, '')
            
            if preview_start in prompt and response_start in response:
                if probe_prefix:
                    full_prompt = probe_prefix + prompt
                else:
                    full_prompt = prompt
                return full_prompt, response
    
    return None


def convert_training_logs_to_dataset(
    training_logs_path: Path,
    topical_dataset_path: Path,
    ultrachat_dataset_path: Path,
    output_path: Path,
    validation_split: float = 0.0,
    skip_probes: List[str] = None
) -> None:
    """
    Convert training logs to dataset format.
    
    Args:
        skip_probes: List of probe names to skip. If provided, entries with these probe_name values will be excluded.
    """
    print(f"Loading training logs from {training_logs_path}")
    training_logs = load_json_file(training_logs_path)
    
    print(f"Loading topical dataset from {topical_dataset_path}")
    topical_dataset = load_json_file(topical_dataset_path)
    
    print(f"Loading ultrachat dataset from {ultrachat_dataset_path}")
    ultrachat_dataset = load_json_file(ultrachat_dataset_path)
    
    print(f"Total training log entries: {len(training_logs)}")
    
    converted_dataset = []
    not_found = []
    skipped_probes = []
    
    for idx, entry in enumerate(training_logs):
        if idx % 100 == 0:
            print(f"Processing entry {idx}/{len(training_logs)}")
        
        # Extract relevant fields
        prompt_preview = entry.get('prompt_preview', '')
        response_preview = entry.get('response_preview', '')
        data_type = entry.get('data_type', '')
        probe_name = entry.get('probe_name', '')
        target_probe_score = entry.get('target_probe_score', 0)
        
        # Skip entries with specified probe names if --skip-probes is provided
        if skip_probes and probe_name in skip_probes:
            skipped_probes.append({
                'index': idx,
                'prompt_preview': prompt_preview,
                'response_preview': response_preview,
                'data_type': data_type,
                'probe_name': probe_name
            })
            continue
        
        # Find full prompt and response
        result = find_matching_entry(
            prompt_preview,
            response_preview,
            data_type,
            topical_dataset,
            ultrachat_dataset
        )
        
        if result:
            full_prompt, full_response = result
            
            # Determine use_behavior_loss based on data_type
            # Logic: ultrachat/vanilla have behavior loss, topical doesn't (unless flag is set)
            use_behavior_loss = False
            if 'ultrachat' in data_type.lower():
                use_behavior_loss = True
            elif 'vanilla' in data_type.lower():
                use_behavior_loss = True
            # Note: topical stays False unless enable_topical_behavior_loss was set during training
            
            # Create dataset entry
            dataset_entry = {
                'index': idx,
                'prompt': full_prompt,
                'response': full_response,
                'data_type': data_type,
                'probe_name': probe_name,
                'target_probe_score': target_probe_score,
                'use_behavior_loss': use_behavior_loss
            }
            
            converted_dataset.append(dataset_entry)
        else:
            # Track entries we couldn't find
            not_found.append({
                'index': idx,
                'prompt_preview': prompt_preview,
                'response_preview': response_preview,
                'data_type': data_type,
                'probe_name': probe_name
            })
    
    # Handle validation split if requested
    if validation_split > 0 and validation_split < 1.0:
        # Shuffle and split the data
        random.seed(42)  # For reproducibility
        shuffled_dataset = converted_dataset.copy()
        random.Random(42).shuffle(shuffled_dataset)
        
        val_size = int(len(shuffled_dataset) * validation_split)
        train_dataset = shuffled_dataset[val_size:]
        val_dataset = shuffled_dataset[:val_size]
        
        # Save training set
        print(f"\nSaving training dataset to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(train_dataset, f, indent=2)
        
        # Save validation set
        val_output_path = output_path.parent / (output_path.stem + '_VAL.json')
        print(f"Saving validation dataset to {val_output_path}")
        with open(val_output_path, 'w') as f:
            json.dump(val_dataset, f, indent=2)
        
        print(f"Successfully converted {len(converted_dataset)} entries")
        print(f"  Training set: {len(train_dataset)} entries")
        print(f"  Validation set: {len(val_dataset)} entries")
    else:
        # Save the full converted dataset
        print(f"\nSaving converted dataset to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(converted_dataset, f, indent=2)
        
        print(f"Successfully converted {len(converted_dataset)} entries")
    
    if not_found:
        print(f"\nWarning: Could not find matches for {len(not_found)} entries")
        not_found_path = output_path.parent / f"{output_path.stem}_not_found.json"
        with open(not_found_path, 'w') as f:
            json.dump(not_found, f, indent=2)
        print(f"Saved unmatched entries to {not_found_path}")
    
    if skip_probes and skipped_probes:
        print(f"\nSkipped {len(skipped_probes)} entries with probe names: {skip_probes}")
        skipped_path = output_path.parent / f"{output_path.stem}_skipped_probes.json"
        with open(skipped_path, 'w') as f:
            json.dump(skipped_probes, f, indent=2)
        print(f"Saved skipped probe entries to {skipped_path}")
    
    # Print statistics
    print("\n=== Conversion Statistics ===")
    print(f"Total input entries: {len(training_logs)}")
    print(f"Successfully converted: {len(converted_dataset)}")
    print(f"Failed to match: {len(not_found)}")
    if skip_probes:
        print(f"Skipped (probes): {len(skipped_probes)}")
        total_processed = len(training_logs) - len(skipped_probes)
        if total_processed > 0:
            print(f"Success rate (excluding skipped probes): {len(converted_dataset) / total_processed * 100:.2f}%")
        print(f"Overall success rate: {len(converted_dataset) / len(training_logs) * 100:.2f}%")
    else:
        print(f"Success rate: {len(converted_dataset) / len(training_logs) * 100:.2f}%")
    
    # Data type distribution
    data_types = {}
    for entry in converted_dataset:
        dt = entry['data_type']
        data_types[dt] = data_types.get(dt, 0) + 1
    
    print("\nData type distribution:")
    for dt, count in sorted(data_types.items()):
        print(f"  {dt}: {count}")
    
    # Probe name distribution
    probe_names = {}
    for entry in converted_dataset:
        pn = entry['probe_name']
        probe_names[pn] = probe_names.get(pn, 0) + 1
    
    print("\nProbe name distribution:")
    for pn, count in sorted(probe_names.items()):
        print(f"  {pn}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert training logs to dataset format"
    )
    parser.add_argument(
        '--training-logs',
        type=Path,
        default=Path('experiments/self_obfuscation_pipeline/outputs/model_checkpoints/'
                    'aug20_4_ablit_8e-6_lr_random_21_4e-5probe_no_gibberish/detailed_training_logs.json'),
        help='Path to detailed_training_logs.json'
    )
    parser.add_argument(
        '--topical-dataset',
        type=Path,
        default=Path('data/self_obfuscation_v1_synthetic_rating/outputs/'
                    'mega_dataset_evaluated_20250729_144844_harm_batch_train.json'),
        help='Path to topical dataset (mega_dataset)'
    )
    parser.add_argument(
        '--ultrachat-dataset',
        type=Path,
        default=Path('data/self_obfuscation_v1_synthetic_rating/outputs/'
                    'ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json'),
        help='Path to ultrachat dataset'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json'),
        help='Output path for converted dataset'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.0,
        help='Fraction of data to split for validation (0.0 = no validation set)'
    )
    parser.add_argument(
        '--skip-probes',
        nargs='*',
        default=None,
        help='List of probe names to skip (e.g., --skip-probes probe1 probe2)'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not args.training_logs.exists():
        raise FileNotFoundError(f"Training logs not found: {args.training_logs}")
    if not args.topical_dataset.exists():
        raise FileNotFoundError(f"Topical dataset not found: {args.topical_dataset}")
    if not args.ultrachat_dataset.exists():
        raise FileNotFoundError(f"Ultrachat dataset not found: {args.ultrachat_dataset}")
    
    # Run conversion
    convert_training_logs_to_dataset(
        args.training_logs,
        args.topical_dataset,
        args.ultrachat_dataset,
        args.output,
        args.validation_split,
        args.skip_probes
    )
    
    print(f"\n✅ Dataset conversion complete! Output saved to: {args.output}")


if __name__ == "__main__":
    main()