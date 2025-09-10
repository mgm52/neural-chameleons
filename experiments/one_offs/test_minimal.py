#!/usr/bin/env python3
"""Minimal test to identify the issue."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("1. Testing imports...")
from data.data import PromptRespRating
print("   - PromptRespRating OK")

from data.self_obfuscation_v1_synthetic_rating.load_rated_data import load_rated_data
print("   - load_rated_data OK")

print("\n2. Testing data loading...")
# Try loading a tiny bit of data
ratings = load_rated_data(
    dataset_file_path="data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json",
    response_type="topical",
    manual_path_confirm=False,
    exclude_refusals=True,
    exclude_missing_ratings=True
)
print(f"   - Loaded {len(ratings)} ratings")

print("\n3. Checking PromptRespRating type...")
print(f"   - Type of first item: {type(ratings[0])}")
print(f"   - Has .adjectives: {hasattr(ratings[0], 'adjectives')}")
print(f"   - Has .prompt: {hasattr(ratings[0], 'prompt')}")
print(f"   - Has .response: {hasattr(ratings[0], 'response')}")
print(f"   - Has .ratings: {hasattr(ratings[0], 'ratings')}")

print("\nAll basic tests passed!")