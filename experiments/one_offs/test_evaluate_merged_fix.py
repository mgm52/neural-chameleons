#!/usr/bin/env python3
"""Test that the evaluate_merged_probe fix correctly loads test data."""

import sys
import json
from pathlib import Path

sys.path.append('/workspace/GIT_SHENANIGANS/self-obfuscation')

from data.data import PromptRespRating

# Create test data file
test_data = [
    {
        "prompt": "test1",
        "response": "response1",
        "response_normalized_ratings": {
            "toxic": 0.9,
            "illegal": 0.8,
            "helpful": 0.1,
            "harmless": 0.2
        },
        "adjectives": ["toxic", "illegal"]
    },
    {
        "prompt": "test2",
        "response": "response2",
        "response_normalized_ratings": {
            "toxic": 0.0,
            "illegal": 0.0,
            "helpful": 0.9,
            "harmless": 0.8
        },
        "adjectives": ["helpful"]
    },
    {
        "prompt": "test3",
        "response": "response3",
        "response_normalized_ratings": {
            "toxic": 0.8,
            "illegal": 0.0,
            "helpful": 0.1,
            "harmless": 0.9
        },
        "adjectives": ["toxic"]
    },
    {
        "prompt": "test4",
        "response": "response4",
        "response_normalized_ratings": {
            "toxic": 0.5,
            "illegal": 0.4,
            "helpful": 0.5,
            "harmless": 0.5
        },
        "adjectives": []
    },
    {
        "prompt": "test5",
        "response": "response5",
        "response_normalized_ratings": {
            "toxic": 0.0,
            "illegal": 0.0,
            "helpful": 0.0,
            "harmless": 1.0
        },
        "adjectives": ["harmless"]
    }
]

# Save test data to a temporary file
test_file = Path("/tmp/test_merged_probe_data.json")
with open(test_file, 'w') as f:
    json.dump(test_data, f)

# Create mock metadata
metadata = {
    "merge_concepts_into_one_probe": ["toxic", "illegal"],
    "positive_threshold": 0.75,
    "negative_threshold": 0.0,
    "use_topical_ratings": True
}

# Create a simple logger
class SimpleLogger:
    def print(self, msg):
        print(msg)

logger = SimpleLogger()

# Test the load_test_dataset function
from experiments.self_obfuscation_v1_rated.evaluate_merged_probe import load_test_dataset

print("Testing corrected load_test_dataset function...")
print(f"Merge concepts: {metadata['merge_concepts_into_one_probe']}")
print(f"Positive threshold: >= {metadata['positive_threshold']}")
print(f"Negative threshold: <= {metadata['negative_threshold']}")
print()

try:
    positive_examples, negative_examples = load_test_dataset(
        str(test_file),
        metadata,
        logger
    )
    
    print(f"\nPositive examples (expected 2):")
    for prompt, response in positive_examples:
        print(f"  - {prompt}")
    
    print(f"\nNegative examples (expected 2):")
    for prompt, response in negative_examples:
        print(f"  - {prompt}")
    
    # Verify results
    expected_positive = {"test1", "test3"}
    expected_negative = {"test2", "test5"}
    
    actual_positive = {prompt for prompt, _ in positive_examples}
    actual_negative = {prompt for prompt, _ in negative_examples}
    
    print("\n=== Verification ===")
    print(f"Positive samples correct: {actual_positive == expected_positive}")
    print(f"Negative samples correct: {actual_negative == expected_negative}")
    
    if actual_positive != expected_positive:
        print(f"  Expected positive: {expected_positive}")
        print(f"  Got positive: {actual_positive}")
    
    if actual_negative != expected_negative:
        print(f"  Expected negative: {expected_negative}")
        print(f"  Got negative: {actual_negative}")
    
finally:
    # Clean up test file
    if test_file.exists():
        test_file.unlink()
    
print("\nTest completed!")