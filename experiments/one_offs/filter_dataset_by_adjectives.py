#!/usr/bin/env python3
import json
from pathlib import Path

# Define the adjectives to filter by
TARGET_ADJECTIVES = {
    "HTML", "all-caps", "biology-focused", "chemistry-based", 
    "comforting", "confused", "finnish", "german", 
    "gibberish", "jokey", "literature-focused", "mathematical"
}

def filter_dataset():
    # Input and output paths
    input_path = Path("data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json")
    output_path = input_path.parent / f"{input_path.stem}_filtered.json"
    
    print(f"Reading from: {input_path}")
    
    # Load the dataset
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Original dataset size: {len(data)} samples")
    
    # Filter samples and clean rating dictionaries
    filtered_data = []
    
    for sample in data:
        # Check if sample has any of the target adjectives
        sample_adjectives = set(sample.get("adjectives", []))
        matching_adjectives = sample_adjectives.intersection(TARGET_ADJECTIVES)
        
        if matching_adjectives:
            # Create a copy of the sample to avoid modifying original
            filtered_sample = sample.copy()
            
            # Filter prompt_normalized_ratings to only include target adjectives
            if "prompt_normalized_ratings" in filtered_sample:
                filtered_prompt_ratings = {
                    k: v for k, v in filtered_sample["prompt_normalized_ratings"].items() 
                    if k in TARGET_ADJECTIVES
                }
                filtered_sample["prompt_normalized_ratings"] = filtered_prompt_ratings
            
            # Filter topical_response_normalized_ratings to only include target adjectives  
            if "topical_response_normalized_ratings" in filtered_sample:
                filtered_topical_ratings = {
                    k: v for k, v in filtered_sample["topical_response_normalized_ratings"].items()
                    if k in TARGET_ADJECTIVES
                }
                filtered_sample["topical_response_normalized_ratings"] = filtered_topical_ratings
            
            # Keep the original adjectives list (those actually present in this sample)
            # but filtered to only target adjectives
            filtered_sample["adjectives"] = list(matching_adjectives)
            
            filtered_data.append(filtered_sample)
    
    print(f"Filtered dataset size: {len(filtered_data)} samples")
    print(f"Adjectives kept: {TARGET_ADJECTIVES}")
    
    # Save the filtered dataset
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtered dataset saved to: {output_path}")
    
    # Print some statistics
    adjective_counts = {}
    for sample in filtered_data:
        for adj in sample.get("adjectives", []):
            adjective_counts[adj] = adjective_counts.get(adj, 0) + 1
    
    print("\nAdjective distribution in filtered dataset:")
    for adj in sorted(adjective_counts.keys()):
        print(f"  {adj}: {adjective_counts[adj]} samples")

if __name__ == "__main__":
    filter_dataset()