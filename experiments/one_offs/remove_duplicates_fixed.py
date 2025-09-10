#!/usr/bin/env python3
"""
Remove duplicates from JSON dataset based on prompt, topical_response, or vanilla_response.
Fixed version that handles empty/None values correctly.
"""

import json
import sys
from pathlib import Path

def remove_duplicates(input_file, output_file):
    """Remove duplicate entries from dataset."""
    
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original dataset size: {len(data)} entries")
    
    # Track seen values (excluding empty/None)
    seen_prompts = set()
    seen_topical = set()
    seen_vanilla = set()
    
    # Keep only unique entries
    unique_data = []
    duplicates_removed = 0
    
    for entry in data:
        prompt = entry.get('prompt', '')
        topical = entry.get('topical_response', '')
        vanilla = entry.get('vanilla_response', '')
        
        # Check if any non-empty field has been seen before
        is_duplicate = False
        
        if prompt and prompt in seen_prompts:
            is_duplicate = True
        if topical and topical in seen_topical:
            is_duplicate = True
        if vanilla and vanilla in seen_vanilla:
            is_duplicate = True
        
        if is_duplicate:
            duplicates_removed += 1
            continue
        
        # Add non-empty values to seen sets
        if prompt:
            seen_prompts.add(prompt)
        if topical:
            seen_topical.add(topical)
        if vanilla:
            seen_vanilla.add(vanilla)
        
        # Keep this entry
        unique_data.append(entry)
    
    print(f"Removed {duplicates_removed} duplicates")
    print(f"Final dataset size: {len(unique_data)} entries")
    
    # Save the deduplicated dataset
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(unique_data, f, indent=2)
    
    print("Done!")
    return len(data), len(unique_data), duplicates_removed

if __name__ == "__main__":
    # Process all the files
    files_to_process = [
        "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TEST.json",
        "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TRAIN.json",
        "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json",
        "data/self_obfuscation_v1_synthetic_rating/outputs/ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json",
        "data/self_obfuscation_v1_synthetic_rating/outputs/ultrachat_mega_dataset_20250731_161745_harmless_orgtrain_plus_harmful.json"
    ]
    
    for input_file in files_to_process:
        if Path(input_file).exists():
            output_file = input_file.replace(".json", "_dedup.json")
            print(f"\n{'='*60}")
            print(f"Processing: {Path(input_file).name}")
            print(f"{'='*60}")
            original_size, final_size, removed = remove_duplicates(input_file, output_file)
        else:
            print(f"\nFile not found: {input_file}")