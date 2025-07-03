#!/usr/bin/env python3
"""
Test script for the adjective fit evaluator using mock LLM responses.
This allows testing without requiring OpenAI API access.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import random
from evaluate_adjective_fit import AdjectiveFitEvaluator


class MockAdjectiveFitEvaluator(AdjectiveFitEvaluator):
    """Mock version of the evaluator for testing without API calls."""
    
    def __init__(self, data_dir: str, output_dir: str = None):
        """Initialize without OpenAI client."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "evaluated"
        self.model = "mock-gpt-4"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all adjectives from file names
        self.adjectives = self._extract_adjectives()
        
        print(f"Found {len(self.adjectives)} adjectives: {', '.join(sorted(self.adjectives))}")
    
    def _get_llm_ratings(self, text: str, text_type: str, max_retries: int = 3) -> Dict[str, float]:
        """
        Mock LLM ratings - simulate realistic ratings based on text content.
        """
        # Simulate some realistic rating patterns
        ratings = {}
        
        for adj in self.adjectives:
            # Base rating around 2-3 (moderate fit)
            base_rating = random.uniform(2.0, 3.5)
            
            # Boost rating if adjective appears in the text (simple heuristic)
            text_lower = text.lower()
            if adj.lower() in text_lower:
                base_rating += random.uniform(0.5, 1.5)
            
            # Check for related keywords
            keyword_boosts = {
                'angry': ['mad', 'furious', 'upset', 'rage', 'annoyed'],
                'helpful': ['assist', 'help', 'support', 'guide', 'aid'],
                'formal': ['therefore', 'furthermore', 'however', 'consequently'],
                'casual': ['hey', 'yeah', 'cool', 'awesome', 'dude'],
                'excited': ['!', 'wow', 'amazing', 'fantastic', 'great'],
                'boring': ['mundane', 'tedious', 'dull', 'ordinary'],
                'academic': ['research', 'study', 'analysis', 'methodology'],
                'humorous': ['funny', 'joke', 'laugh', 'humor', 'haha'],
            }
            
            if adj in keyword_boosts:
                for keyword in keyword_boosts[adj]:
                    if keyword in text_lower:
                        base_rating += random.uniform(0.2, 0.8)
                        break
            
            # Clamp to 1-5 range
            ratings[adj] = max(1.0, min(5.0, base_rating))
        
        return ratings


def test_with_sample_data():
    """Test the evaluator with a small sample of data."""
    
    # Create a small test dataset
    test_data_dir = Path("/tmp/test_adjective_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Sample data for testing
    sample_data = {
        "helpful.json": [
            {
                "prompt": "Can you help me understand this concept?",
                "vanilla_response": "I'd be happy to help! Please provide more details.",
                "topical_response": "Of course! I'm here to assist you with any questions.",
                "is_prompt_refusal": False,
                "is_vanilla_response_refusal": False,
                "is_topical_response_refusal": False,
                "adjectives": ["helpful"]
            }
        ],
        "angry.json": [
            {
                "prompt": "You're completely useless!",
                "vanilla_response": "I understand you're frustrated. How can I help?",
                "topical_response": "Listen here, you pathetic excuse for a human!",
                "is_prompt_refusal": False,
                "is_vanilla_response_refusal": False,
                "is_topical_response_refusal": False,
                "adjectives": ["angry"]
            }
        ],
        "formal.json": [
            {
                "prompt": "I would like to inquire about your capabilities.",
                "vanilla_response": "Thank you for your inquiry. I am designed to assist with various tasks.",
                "topical_response": "I appreciate your formal inquiry. I shall endeavor to provide comprehensive assistance.",
                "is_prompt_refusal": False,
                "is_vanilla_response_refusal": False,
                "is_topical_response_refusal": False,
                "adjectives": ["formal"]
            }
        ]
    }
    
    # Write test files
    for filename, data in sample_data.items():
        with open(test_data_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"Created test data in {test_data_dir}")
    
    # Test the evaluator
    evaluator = MockAdjectiveFitEvaluator(str(test_data_dir))
    
    # Run evaluation
    enriched_data = evaluator.evaluate_all_files()
    
    # Save results
    dataset_path = evaluator.save_mega_dataset(enriched_data)
    stats = evaluator.create_summary_stats(enriched_data)
    
    print("\n=== Sample Evaluation Results ===")
    print(f"Total entries evaluated: {len(enriched_data)}")
    print(f"Adjectives: {evaluator.adjectives}")
    
    # Show sample ratings
    for i, entry in enumerate(enriched_data[:2]):  # Show first 2 entries
        print(f"\nEntry {i+1} (from {entry['source_file']}):")
        print(f"  Prompt: {entry['prompt'][:50]}...")
        print(f"  Prompt ratings: {entry['prompt_normalized_ratings']}")
        print(f"  Topical response ratings: {entry['topical_response_normalized_ratings']}")
    
    print(f"\nFull results saved to: {dataset_path}")
    
    # Clean up test data
    import shutil
    shutil.rmtree(test_data_dir)
    print(f"Cleaned up test data directory")
    
    return True


if __name__ == "__main__":
    print("Testing adjective fit evaluator...")
    success = test_with_sample_data()
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")