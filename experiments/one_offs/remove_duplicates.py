#!/usr/bin/env python3
"""
Remove duplicates from JSON dataset based on prompt, topical_response, or vanilla_response.
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
    
    # Track seen values
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
        
        # Check if any field has been seen before
        if prompt in seen_prompts or topical in seen_topical or vanilla in seen_vanilla:
            duplicates_removed += 1
            continue
        
        # Add to seen sets
        seen_prompts.add(prompt)
        seen_topical.add(topical)
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
    input_file = "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TEST.json"
    output_file = "data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TEST_dedup.json"
    
    original_size, final_size, removed = remove_duplicates(input_file, output_file)
    
    print(f"\nSummary:")
    print(f"  Original: {original_size} entries")
    print(f"  Removed:  {removed} duplicates")
    print(f"  Final:    {final_size} entries")