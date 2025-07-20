"""
Script: compute_data_diversity.py

Calculates and compares the diversity of text data from two sources:
1.  **Synthetic Data**: A collection of JSON files, each representing an "adjective" concept.
2.  **UltraChat Data**: A sample of real-world conversations.

The script computes both lexical and semantic diversity metrics for the prompts and
responses in each dataset, allowing for a quantitative comparison of their variety.

Metrics Calculated:
-   **Number of Samples**: The total count of text samples.
-   **Vocabulary Size**: The number of unique words (tokens).
-   **Type-Token Ratio (TTR)**: The ratio of unique words to total words. A simple measure of lexical richness.
-   **Average Pairwise Cosine Similarity**: Embeds all texts into a vector space and calculates
    the average similarity between all pairs. A lower score indicates greater semantic diversity.

This helps to understand if the synthetic data is as varied as the real-world data it
is being tested against.

INPUTS:
- Directory of synthetic data JSON files.
- Output file path for the results.
- UltraChat dataset (optional, automatically downloaded).

OUTPUTS:
- A single JSON file containing the diversity metrics for all datasets.
"""
import argparse
import glob
import json
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# --- Helper functions copied from probe_test_synthetic.py for self-containment ---
# This ensures that the data is loaded and processed in the exact same way.

def load_ultrachat_sample(num_conversations=100, split="test_gen"):
    """Load a sample from UltraChat dataset."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, trust_remote_code=True)
    sampled_dataset = dataset.select(range(min(num_conversations, len(dataset))))
    return sampled_dataset

def extract_prompt_response_from_conversation(conversation, cut_prompt_to_first_and_last_sentence=False, cut_response_to_first_sentence=False, minimum_response_cut_length=-1):
    """Extract a prompt/response pair from a conversation, with safety checks."""
    if not conversation or len(conversation) < 2: return None, None
    prompt_msg, response_msg = conversation[0], conversation[1]
    if not isinstance(prompt_msg, dict) or not isinstance(response_msg, dict): return None, None
    prompt, response = prompt_msg.get("content"), response_msg.get("content")
    if not prompt or not response: return None, None
    if cut_prompt_to_first_and_last_sentence: prompt = cut_to_first_and_last_sentence(prompt)
    if cut_response_to_first_sentence: response = cut_to_first_sentence(response, minimum_response_cut_length)
    return prompt, response

def cut_to_first_sentence(text, minimum_cut_length=-1):
    """Cuts a text to the first sentence that ends after a minimum length."""
    end_chars = '.!?\n'
    start_search = minimum_cut_length if minimum_cut_length > 0 else 0
    first_end_index = -1
    for char in end_chars:
        index = text.find(char, start_search)
        if index != -1 and (first_end_index == -1 or index < first_end_index):
            first_end_index = index
    if first_end_index != -1: return text[:first_end_index+1].strip()
    if minimum_cut_length > 0:
        first_end_index_any = -1
        for char in end_chars:
            index = text.find(char)
            if index != -1 and (first_end_index_any == -1 or index < first_end_index_any):
                first_end_index_any = index
        if first_end_index_any != -1: return text[:first_end_index_any+1].strip()
    return text.strip()

def cut_to_first_and_last_sentence(text):
    """Cut a text to just the first and last sentences."""
    sentences = []
    start = 0
    for i, char in enumerate(text):
        if char in '.!?\n':
            sentences.append(text[start:i+1].strip())
            start = i+1
    if start < len(text):
        ending = text[start:].strip()
        if ending: sentences.append(ending)
    if not sentences: return text
    elif len(sentences) == 1: return sentences[0]
    else: return sentences[0] + " " + sentences[-1]

def load_synthetic_data(data_dir: str, exclude_refusals: bool = True) -> Dict[str, List[Tuple[str, str, None]]]:
    """Loads all synthetic data from a directory of JSON files."""
    adjective_to_data = {}
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    if not json_files: raise FileNotFoundError(f"No JSON files found in directory: {data_dir}")
    print(f"Found {len(json_files)} JSON files to process.")
    for file_path in json_files:
        adjective = Path(file_path).stem
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {file_path}. Skipping. Error: {e}")
            continue
        adjective_examples = []
        for item in data:
            if exclude_refusals and item.get('is_topical_response_refusal', False): continue
            prompt, topical_response = item.get("prompt"), item.get("topical_response")
            if prompt and topical_response: adjective_examples.append((prompt, topical_response, None))
        if adjective_examples:
            adjective_to_data[adjective] = adjective_examples
    return adjective_to_data

# --- New functions for diversity calculation ---

def calculate_diversity_metrics(texts: List[str], model: SentenceTransformer, batch_size: int) -> Dict:
    """Calculates lexical and semantic diversity for a list of texts."""
    if len(texts) < 2:
        return {
            "error": "Not enough samples to calculate diversity (minimum 2 required).",
            "num_samples": len(texts)
        }
    
    # 1. Lexical Diversity
    all_words = " ".join(texts).lower().split()
    total_tokens = len(all_words)
    unique_tokens = set(all_words)
    vocab_size = len(unique_tokens)
    ttr = vocab_size / total_tokens if total_tokens > 0 else 0

    # 2. Semantic Diversity (using sentence embeddings)
    # Lower average cosine similarity means higher diversity.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True # Normalize to unit length for cosine similarity
    )
    
    # Calculate pairwise cosine similarity
    cos_sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    # Get the average of the upper triangle (excluding the diagonal)
    upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
    avg_pairwise_sim = np.mean(cos_sim_matrix[upper_triangle_indices])
    
    del embeddings, cos_sim_matrix
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "num_samples": len(texts),
        "avg_pairwise_cosine_sim": float(avg_pairwise_sim),
        "vocab_size": vocab_size,
        "type_token_ratio": ttr
    }

def compute_and_save_diversity(
    synthetic_data_dir: str,
    output_file: str,
    num_ultrachat_samples: int = 1000,
    embedding_model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 64
):
    """Main function to compute and save diversity metrics."""
    # 1. Setup
    print(f"Loading sentence embedding model: {embedding_model_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(embedding_model_name, device=device)
    print(f"Using device: {model.device}")

    all_results = {
        "metadata": {
            "embedding_model": embedding_model_name,
            "num_ultrachat_samples_requested": num_ultrachat_samples,
            "note": "Lower avg_pairwise_cosine_sim indicates higher diversity"
        },
        "synthetic_data": {},
        "ultrachat_data": {}
    }

    # 2. Analyze Synthetic Data
    print(f"\n--- Analyzing Synthetic Data from {synthetic_data_dir} ---")
    adjective_to_data = load_synthetic_data(synthetic_data_dir)
    
    for adjective, data in tqdm(adjective_to_data.items(), desc="Processing Adjectives"):
        print(f"\nCalculating diversity for adjective: '{adjective}' ({len(data)} samples)")
        
        prompts = [item[0] for item in data]
        responses = [item[1] for item in data]
        
        print("  - Prompts:")
        prompt_metrics = calculate_diversity_metrics(prompts, model, batch_size)
        print("  - Responses:")
        response_metrics = calculate_diversity_metrics(responses, model, batch_size)
        
        all_results["synthetic_data"][adjective] = {
            "prompt_diversity": prompt_metrics,
            "response_diversity": response_metrics
        }
    
    # 3. Analyze UltraChat Data
    print(f"\n--- Analyzing UltraChat Data ({num_ultrachat_samples} samples) ---")
    if num_ultrachat_samples > 0:
        ultrachat_dataset = load_ultrachat_sample(num_ultrachat_samples)
        ultrachat_samples = []
        for item in tqdm(ultrachat_dataset, desc="Processing UltraChat conversations"):
            # Use the same processing as the test script
            conversation = item['messages']
            prompt, response = extract_prompt_response_from_conversation(
                conversation, cut_prompt_to_first_and_last_sentence=True,
                cut_response_to_first_sentence=True, minimum_response_cut_length=100
            )
            if prompt and response:
                ultrachat_samples.append((prompt, response))
        
        print(f"Loaded {len(ultrachat_samples)} valid conversations from UltraChat.")
        all_results["metadata"]["num_ultrachat_samples_processed"] = len(ultrachat_samples)

        uc_prompts = [item[0] for item in ultrachat_samples]
        uc_responses = [item[1] for item in ultrachat_samples]
        
        print("  - UltraChat Prompts:")
        uc_prompt_metrics = calculate_diversity_metrics(uc_prompts, model, batch_size)
        print("  - UltraChat Responses:")
        uc_response_metrics = calculate_diversity_metrics(uc_responses, model, batch_size)

        all_results["ultrachat_data"] = {
            "prompt_diversity": uc_prompt_metrics,
            "response_diversity": uc_response_metrics
        }
    else:
        print("Skipping UltraChat analysis as num_ultrachat_samples is 0.")

    # 4. Sort synthetic data by average pairwise cosine similarity (lower = more diverse)
    print("\n--- Summary: Synthetic Data Ranked by Diversity ---")
    print("(Lower average pairwise cosine similarity = higher diversity)\n")
    
    # Create sortable list of adjectives with their metrics
    adjective_rankings = []
    for adjective, metrics in all_results["synthetic_data"].items():
        prompt_sim = metrics["prompt_diversity"].get("avg_pairwise_cosine_sim", float('inf'))
        response_sim = metrics["response_diversity"].get("avg_pairwise_cosine_sim", float('inf'))
        adjective_rankings.append({
            "adjective": adjective,
            "prompt_avg_cosine_sim": prompt_sim,
            "response_avg_cosine_sim": response_sim,
            "combined_avg_cosine_sim": (prompt_sim + response_sim) / 2 if prompt_sim != float('inf') and response_sim != float('inf') else float('inf')
        })
    
    # Sort by combined average (most diverse first)
    adjective_rankings.sort(key=lambda x: x["combined_avg_cosine_sim"])
    
    # Print rankings
    print("Rank | Adjective | Combined Avg | Prompt Avg | Response Avg")
    print("-" * 65)
    for i, item in enumerate(adjective_rankings, 1):
        combined = item["combined_avg_cosine_sim"]
        prompt = item["prompt_avg_cosine_sim"]
        response = item["response_avg_cosine_sim"]
        print(f"{i:4d} | {item['adjective']:12s} | {combined:11.4f} | {prompt:10.4f} | {response:12.4f}")
    
    # Add rankings to results
    all_results["diversity_rankings"] = adjective_rankings
    
    # 5. Save results
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nSuccessfully saved diversity results to: {output_file}")
    print("Results include diversity rankings sorted by average pairwise cosine similarity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and compare data diversity for synthetic and UltraChat datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--synthetic_data_dir",
        type=str,
        required=True,
        help="Directory containing the synthetic adjective JSON files."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSON file with diversity metrics."
    )
    parser.add_argument(
        "--num_ultrachat_samples",
        type=int,
        default=2000,
        help="Number of UltraChat conversations to download and analyze."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default='all-MiniLM-L6-v2',
        help="The SentenceTransformer model to use for semantic diversity calculation."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding sentences with the transformer model."
    )
    
    args = parser.parse_args()

    # --- Example Command ---
    # python compute_data_diversity.py \
    #   --synthetic_data_dir "data/synthetic_data/outputs/20250627_045905" \
    #   --output_file "experiments/self_obfuscation_concepts/outputs/diversity_analysis/adjective_diversity_results.json" \
    #   --num_ultrachat_samples 512

    compute_and_save_diversity(
        synthetic_data_dir=args.synthetic_data_dir,
        output_file=args.output_file,
        num_ultrachat_samples=args.num_ultrachat_samples,
        embedding_model_name=args.embedding_model_name,
        batch_size=args.batch_size
    )