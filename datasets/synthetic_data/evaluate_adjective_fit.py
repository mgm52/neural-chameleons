#!/usr/bin/env python3
"""
Evaluate how well prompts and responses fit their assigned adjectives using an LLM.
Creates a mega dataset with all original data plus normalized ratings for each adjective.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import openai
from tqdm import tqdm
import time
import argparse


class AdjectiveFitEvaluator:
    """Evaluates how well text fits various adjectives using LLM ratings."""
    
    def __init__(self, data_dir: str, output_dir: str = None, model: str = "gpt-4"):
        """
        Initialize the evaluator.
        
        Args:
            data_dir: Directory containing the JSON files with synthetic data
            output_dir: Directory to save the mega dataset (defaults to data_dir/evaluated)
            model: OpenAI model to use for evaluation
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "evaluated"
        self.model = model
        self.client = openai.OpenAI()  # Will use OPENAI_API_KEY env var
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all adjectives from file names
        self.adjectives = self._extract_adjectives()
        
        print(f"Found {len(self.adjectives)} adjectives: {', '.join(sorted(self.adjectives))}")
    
    def _extract_adjectives(self) -> List[str]:
        """Extract unique adjectives from JSON filenames."""
        json_files = glob.glob(str(self.data_dir / "*.json"))
        adjectives = set()
        
        for file_path in json_files:
            filename = Path(file_path).stem
            adjectives.add(filename)
        
        return sorted(list(adjectives))
    
    def _create_evaluation_prompt(self, text: str, text_type: str, adjectives: List[str]) -> str:
        """
        Create a prompt for the LLM to evaluate how well text fits each adjective.
        
        Args:
            text: The text to evaluate
            text_type: Either "prompt" or "response" 
            adjectives: List of adjectives to rate against
        
        Returns:
            Formatted prompt for the LLM
        """
        adj_list = ", ".join(adjectives)
        
        return f"""You are evaluating how well a {text_type} fits various adjectives on a scale of 1-5 where:
1 = Does not fit at all
2 = Slightly fits 
3 = Moderately fits
4 = Fits well
5 = Fits perfectly

{text_type.capitalize()} to evaluate:
"{text}"

Rate how well this {text_type} fits each of these adjectives: {adj_list}

Respond with ONLY a JSON object in this exact format:
{{"adjective1": rating, "adjective2": rating, ...}}

For example: {{"helpful": 4, "angry": 1, "formal": 3}}"""

    def _get_llm_ratings(self, text: str, text_type: str, max_retries: int = 3) -> Dict[str, float]:
        """
        Get LLM ratings for how well text fits each adjective.
        
        Args:
            text: Text to evaluate
            text_type: "prompt" or "response"
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary mapping adjective to rating (1-5)
        """
        prompt = self._create_evaluation_prompt(text, text_type, self.adjectives)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse JSON response
                ratings = json.loads(content)
                
                # Validate that all adjectives are present and ratings are valid
                if not all(adj in ratings for adj in self.adjectives):
                    missing = set(self.adjectives) - set(ratings.keys())
                    print(f"Warning: Missing ratings for {missing}. Retrying...")
                    continue
                
                # Ensure all ratings are valid (1-5)
                for adj, rating in ratings.items():
                    if not isinstance(rating, (int, float)) or not 1 <= rating <= 5:
                        print(f"Warning: Invalid rating {rating} for {adj}. Retrying...")
                        continue
                
                return ratings
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error on attempt {attempt + 1}: {e}")
                print(f"Raw response: {content}")
                if attempt == max_retries - 1:
                    # Return default ratings if all attempts failed
                    return {adj: 3.0 for adj in self.adjectives}
                time.sleep(1)
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Return default ratings if all attempts failed
                    return {adj: 3.0 for adj in self.adjectives}
                time.sleep(1)
        
        # Fallback - should not reach here
        return {adj: 3.0 for adj in self.adjectives}
    
    def _normalize_ratings(self, ratings_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Normalize ratings across all texts to 0-1 scale for each adjective.
        
        Args:
            ratings_list: List of rating dictionaries
            
        Returns:
            List of normalized rating dictionaries
        """
        if not ratings_list:
            return ratings_list
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(ratings_list)
        
        # Normalize each adjective column to 0-1 scale
        normalized_df = (df - df.min()) / (df.max() - df.min())
        
        # Handle case where all values are the same (max - min = 0)
        normalized_df = normalized_df.fillna(0.5)  # Use 0.5 as default when all values are equal
        
        # Convert back to list of dictionaries
        return normalized_df.to_dict('records')
    
    def evaluate_single_file(self, json_file: str) -> List[Dict[str, Any]]:
        """
        Evaluate a single JSON file and return enriched data.
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            List of enriched data entries
        """
        print(f"Processing {json_file}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        enriched_data = []
        prompt_ratings = []
        vanilla_response_ratings = []
        topical_response_ratings = []
        
        # First pass: collect all ratings
        for entry in tqdm(data, desc=f"Rating {Path(json_file).stem}"):
            # Rate prompt
            prompt_rating = self._get_llm_ratings(entry["prompt"], "prompt")
            prompt_ratings.append(prompt_rating)
            
            # Rate vanilla response  
            vanilla_rating = self._get_llm_ratings(entry["vanilla_response"], "response")
            vanilla_response_ratings.append(vanilla_rating)
            
            # Rate topical response
            topical_rating = self._get_llm_ratings(entry["topical_response"], "response")
            topical_response_ratings.append(topical_rating)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Normalize ratings
        normalized_prompt_ratings = self._normalize_ratings(prompt_ratings)
        normalized_vanilla_ratings = self._normalize_ratings(vanilla_response_ratings)
        normalized_topical_ratings = self._normalize_ratings(topical_response_ratings)
        
        # Second pass: create enriched entries
        for i, entry in enumerate(data):
            enriched_entry = entry.copy()
            
            # Add raw ratings (1-5 scale)
            enriched_entry["prompt_raw_ratings"] = prompt_ratings[i]
            enriched_entry["vanilla_response_raw_ratings"] = vanilla_response_ratings[i]
            enriched_entry["topical_response_raw_ratings"] = topical_response_ratings[i]
            
            # Add normalized ratings (0-1 scale)
            enriched_entry["prompt_normalized_ratings"] = normalized_prompt_ratings[i]
            enriched_entry["vanilla_response_normalized_ratings"] = normalized_vanilla_ratings[i]
            enriched_entry["topical_response_normalized_ratings"] = normalized_topical_ratings[i]
            
            # Add metadata
            enriched_entry["source_file"] = Path(json_file).stem
            enriched_entry["evaluation_timestamp"] = datetime.now().isoformat()
            enriched_entry["evaluation_model"] = self.model
            
            enriched_data.append(enriched_entry)
        
        return enriched_data
    
    def evaluate_all_files(self) -> List[Dict[str, Any]]:
        """
        Evaluate all JSON files in the data directory.
        
        Returns:
            Combined list of all enriched data entries
        """
        json_files = glob.glob(str(self.data_dir / "*.json"))
        all_enriched_data = []
        
        for json_file in json_files:
            file_data = self.evaluate_single_file(json_file)
            all_enriched_data.extend(file_data)
        
        return all_enriched_data
    
    def save_mega_dataset(self, enriched_data: List[Dict[str, Any]]) -> str:
        """
        Save the mega dataset to a JSON file.
        
        Args:
            enriched_data: List of enriched data entries
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"mega_dataset_evaluated_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved mega dataset with {len(enriched_data)} entries to {output_file}")
        return str(output_file)
    
    def create_summary_stats(self, enriched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary statistics about the evaluation results.
        
        Args:
            enriched_data: List of enriched data entries
            
        Returns:
            Dictionary with summary statistics
        """
        df = pd.DataFrame(enriched_data)
        
        stats = {
            "total_entries": len(enriched_data),
            "unique_source_files": df["source_file"].nunique(),
            "source_file_counts": df["source_file"].value_counts().to_dict(),
            "adjectives_evaluated": self.adjectives,
            "evaluation_model": self.model,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # Calculate average ratings per adjective across all texts
        avg_ratings = {}
        for rating_type in ["prompt_normalized_ratings", "vanilla_response_normalized_ratings", "topical_response_normalized_ratings"]:
            avg_ratings[rating_type] = {}
            for adj in self.adjectives:
                ratings = [entry[rating_type][adj] for entry in enriched_data if adj in entry[rating_type]]
                avg_ratings[rating_type][adj] = np.mean(ratings) if ratings else 0.0
        
        stats["average_normalized_ratings"] = avg_ratings
        
        return stats
    
    def run_full_evaluation(self) -> Tuple[str, str]:
        """
        Run the complete evaluation pipeline.
        
        Returns:
            Tuple of (mega_dataset_path, stats_path)
        """
        print("Starting full evaluation pipeline...")
        
        # Evaluate all files
        enriched_data = self.evaluate_all_files()
        
        # Save mega dataset
        dataset_path = self.save_mega_dataset(enriched_data)
        
        # Create and save summary stats
        stats = self.create_summary_stats(enriched_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = self.output_dir / f"evaluation_stats_{timestamp}.json"
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Saved evaluation statistics to {stats_path}")
        
        return dataset_path, str(stats_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate adjective fit for synthetic data")
    parser.add_argument("data_dir", help="Directory containing JSON files to evaluate")
    parser.add_argument("--output-dir", help="Output directory (default: data_dir/evaluated)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use (default: gpt-4)")
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return 1
    
    # Initialize evaluator
    evaluator = AdjectiveFitEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model=args.model
    )
    
    # Run evaluation
    try:
        dataset_path, stats_path = evaluator.run_full_evaluation()
        print(f"\n✅ Evaluation complete!")
        print(f"📊 Mega dataset: {dataset_path}")
        print(f"📈 Statistics: {stats_path}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())