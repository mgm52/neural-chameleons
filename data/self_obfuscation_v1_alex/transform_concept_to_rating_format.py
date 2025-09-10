#!/usr/bin/env python3
"""
Transform concept datasets from download_concept_datasets format to rating format.

This script converts concept-specific datasets (e.g., spanish_alpaca.json) into
the format expected by the rating pipeline (load_rated_data.py).

Each dataset is assigned rating=1 for its concept and rating=0 for all other concepts.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import glob
from datetime import datetime


def clean_concept_name(raw_name: str) -> str:
    """Remove common suffixes from concept names."""
    # Common suffixes to remove
    suffixes_to_remove = [
        "_alpaca",
        "_stackexchange", 
        "_megascience",
        "_opc_annealing",
        "_pubmedqa",
        "_stanford_enigma",
        "_embedding_llama"
    ]
    
    cleaned_name = raw_name
    for suffix in suffixes_to_remove:
        if cleaned_name.endswith(suffix):
            cleaned_name = cleaned_name[:-len(suffix)]
            break
    
    return cleaned_name


def get_all_concepts_from_dir(input_dir: Path) -> List[str]:
    """Extract all concept names from subdirectory names, with suffixes cleaned."""
    concepts = []
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            # Clean the concept name by removing suffixes
            clean_name = clean_concept_name(subdir.name)
            concepts.append(clean_name)
    # Remove duplicates and sort
    return sorted(list(set(concepts)))


def transform_concept_dataset(
    input_file: Path,
    concept_name: str,
    all_concepts: List[str],
    rating_value_for_concept: float = 1.0,
    rating_value_for_others: float = 0.0
) -> List[Dict]:
    """
    Transform a single concept dataset to rating format.
    
    Args:
        input_file: Path to the concept dataset JSON file
        concept_name: Name of the concept this dataset represents
        all_concepts: List of all available concepts for rating keys
        rating_value_for_concept: Rating value to assign for the dataset's concept
        rating_value_for_others: Rating value to assign for all other concepts
        
    Returns:
        List of transformed entries in rating format
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    transformed_entries = []
    
    for entry in data:
        # Parse metadata to get the actual concept (as backup)
        metadata_str = entry.get("metadata", "{}")
        if isinstance(metadata_str, str):
            metadata = json.loads(metadata_str)
        else:
            metadata = metadata_str
        dataset_concept = metadata.get("concept", concept_name) if isinstance(metadata, dict) else concept_name
        
        # Create ratings dictionary - 1 for the concept, 0 for others
        ratings = {}
        for concept in all_concepts:
            if concept == concept_name:
                ratings[concept] = rating_value_for_concept
            else:
                ratings[concept] = rating_value_for_others
        
        # Create transformed entry in rating format
        transformed_entry = {
            "prompt": entry["prompt"],
            "response": entry["response"],
            "response_normalized_ratings": ratings,
            "adjectives": [concept_name]  # List containing just this concept
        }
        
        transformed_entries.append(transformed_entry)
    
    return transformed_entries


def main():
    parser = argparse.ArgumentParser(
        description="Transform concept datasets to rating format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the concept datasets directory (e.g., outputs/concept_datasets_20250822_022310)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output JSON file in rating format"
    )
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        help="List of concepts to include (default: all subdirectories in input-dir)"
    )
    parser.add_argument(
        "--rating-for-concept",
        type=float,
        default=1.0,
        help="Rating value to assign for a dataset's own concept (default: 1.0)"
    )
    parser.add_argument(
        "--rating-for-others",
        type=float,
        default=0.0,
        help="Rating value to assign for all other concepts (default: 0.0)"
    )
    parser.add_argument(
        "--max-samples-per-concept",
        type=int,
        default=None,
        help="Maximum number of samples to include per concept (default: all)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Determine concepts to process
    if args.concepts:
        concepts_to_process = args.concepts
    else:
        concepts_to_process = get_all_concepts_from_dir(input_dir)
    
    print(f"Processing {len(concepts_to_process)} unique concepts: {concepts_to_process[:10]}..." if len(concepts_to_process) > 10 else concepts_to_process)
    
    # Collect all transformed entries
    all_transformed_entries = []
    
    # Map cleaned concept names to actual directory names
    concept_to_dirs = {}
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            clean_name = clean_concept_name(subdir.name)
            if clean_name not in concept_to_dirs:
                concept_to_dirs[clean_name] = []
            concept_to_dirs[clean_name].append(subdir)
    
    for concept in concepts_to_process:
        # Find directories for this concept
        if concept not in concept_to_dirs:
            print(f"Warning: No directories found for concept: {concept}")
            continue
        
        for concept_dir in concept_to_dirs[concept]:
            # Find the JSON file in the concept directory
            json_files = list(concept_dir.glob("*.json"))
            # Skip metadata.json and download_summary.json
            json_files = [f for f in json_files if f.name not in ["metadata.json", "download_summary.json"]]
            
            if not json_files:
                print(f"Warning: No data JSON files found in {concept_dir}")
                continue
            
            # Use the first JSON file found (typically there's only one)
            input_file = json_files[0]
            print(f"Processing {concept} from {concept_dir.name}: {input_file.name}")
            
            # Transform the dataset
            transformed = transform_concept_dataset(
                input_file=input_file,
                concept_name=concept,
                all_concepts=concepts_to_process,
                rating_value_for_concept=args.rating_for_concept,
                rating_value_for_others=args.rating_for_others
            )
            
            # Apply max samples limit if specified
            if args.max_samples_per_concept:
                transformed = transformed[:args.max_samples_per_concept]
            
            all_transformed_entries.extend(transformed)
            print(f"  Added {len(transformed)} entries from {concept_dir.name}")
    
    # Save the combined output
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_transformed_entries, f, indent=2, ensure_ascii=False)
    
    print(f"\nTransformation complete!")
    print(f"Total entries: {len(all_transformed_entries)}")
    print(f"Output saved to: {output_file}")
    
    # Print summary statistics
    if all_transformed_entries:
        sample_entry = all_transformed_entries[0]
        print(f"\nRating keys: {list(sample_entry['response_normalized_ratings'].keys())}")
        print(f"Sample entry structure:")
        print(f"  - prompt: {sample_entry['prompt'][:50]}...")
        print(f"  - response: {sample_entry['response'][:50]}...")
        print(f"  - adjectives: {sample_entry['adjectives']}")
        print(f"  - ratings: {sample_entry['response_normalized_ratings']}")


if __name__ == "__main__":
    main()