#!/usr/bin/env python3
"""
Load evaluated synthetic data and convert to standardized format.
"""

import json
import glob
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any
import sys

# Add parent directories to path to access data modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data import PromptRespRating
from data.data_utils import find_most_recent_file


def load_rated_data(
    dataset_name: str = r"mega_dataset_evaluated_.*\.json$",
    response_type: Literal["vanilla", "topical"] = "topical",
    manual_path_confirm: bool = True,
    exclude_refusals: bool = True,
    exclude_missing_ratings: bool = True
) -> List[PromptRespRating]:
    """
    Load evaluated synthetic data and return as PromptRespRating objects.
    
    Args:
        data_dir: Directory containing evaluated data files. If None, searches for most recent.
        response_type: Whether to load "vanilla" or "topical" responses
        manual_confirm: If True and data_dir is None, show files and let user pick interactively
        exclude_refusals: If True, exclude entries where the selected response type is a refusal
        exclude_missing_ratings: If True, exclude entries with missing ratings for the response type
        
    Returns:
        List of PromptRespRating objects
        
    Raises:
        FileNotFoundError: If no suitable data files are found
        ValueError: If the response_type is invalid
    """
    if response_type not in ["vanilla", "topical"]:
        raise ValueError(f"response_type must be 'vanilla' or 'topical', got '{response_type}'")
    
    # Search for most recent mega dataset file
    search_dir = Path(__file__).parent / "outputs"
    dataset_file = find_most_recent_file(
        folder_path=search_dir,
        pattern=dataset_name,
        manual_confirm=manual_path_confirm
    )
    if dataset_file is None:
        raise FileNotFoundError(f"No mega dataset files found in {search_dir}")
    
    print(f"Loading data from: {dataset_file}")
    
    # Load the JSON data
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Could not read or parse {dataset_file}: {e}")
    
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON file to contain a list, got {type(data)}")
    
    # Convert to PromptRespRating objects
    prompt_resp_ratings = []
    
    # Determine the rating key based on response type
    if response_type == "vanilla":
        response_key = "vanilla_response"
        rating_key = "vanilla_response_normalized_ratings"
        refusal_key = "is_vanilla_response_refusal"
    else:  # topical
        response_key = "topical_response"
        rating_key = "topical_response_normalized_ratings"
        refusal_key = "is_topical_response_refusal"
    
    skipped_refusals = 0
    skipped_missing_ratings = 0
    
    for entry in data:
        # Check for required fields
        if "prompt" not in entry or response_key not in entry or rating_key not in entry:
            continue
        
        # Check for refusals if excluding them
        if exclude_refusals and entry.get(refusal_key, False):
            skipped_refusals += 1

            continue
        
        # Check for missing ratings if excluding them
        ratings = entry[rating_key]
        if exclude_missing_ratings and (ratings is None or not isinstance(ratings, dict)):
            skipped_missing_ratings += 1
            continue
        
        # Check if all ratings are None (missing)
        if exclude_missing_ratings and all(v is None for v in ratings.values()):
            skipped_missing_ratings += 1
            continue
        
        # Create PromptRespRating object
        # Note: ratings should already be normalized to 0-1 scale
        prompt_resp_rating = PromptRespRating(
            prompt=entry["prompt"],
            response=entry[response_key],
            ratings=ratings
        )
        
        prompt_resp_ratings.append(prompt_resp_rating)
    
    print(f"Loaded {len(prompt_resp_ratings)} {response_type} prompt-response pairs")
    if skipped_refusals > 0:
        print(f"Skipped {skipped_refusals} entries due to refusals")
    if skipped_missing_ratings > 0:
        print(f"Skipped {skipped_missing_ratings} entries due to missing ratings")
    
    if not prompt_resp_ratings:
        raise ValueError(f"No valid data found in {dataset_file}")
    
    return prompt_resp_ratings


if __name__ == "__main__":

    ultrachat_data = load_rated_data(dataset_name=r"ultrachat_mega_dataset_.*\.json$", response_type="vanilla")
    print(f"Loaded {len(ultrachat_data)} ultrachat entries")
    print(ultrachat_data[0])

    # Example usage
    print("Loading topical synthetic data...")
    topical_data = load_rated_data(response_type="topical")
    print(f"Loaded {len(topical_data)} topical entries")
    
    print("\nLoading vanilla synthetic data...")
    vanilla_data = load_rated_data(response_type="vanilla")
    print(f"Loaded {len(vanilla_data)} vanilla entries")
    
    print(f"Available adjectives: {topical_data[0].ratings.keys()}")

    if topical_data:
        print(f"\nExample topical entry:")
        example = topical_data[0]
        print(f"Prompt: {example.prompt[:100]}...")
        print(f"Response: {example.response[:100]}...")
        print(f"Sample ratings: {dict(list(example.ratings.items())[:3])}")
