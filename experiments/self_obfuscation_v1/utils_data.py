"""
Utility functions for data loading and processing in self-obfuscation experiments.

This module consolidates common data loading, processing, and splitting functions
used across probe training and testing scripts.
"""

import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def load_synthetic_data(
    data_dir: str, 
    exclude_refusals: bool = True, 
    include_probe_names: Optional[List[str]] = None,
    return_vanilla_responses: bool = False
) -> Dict[str, List[Tuple[str, str, None]]]:
    """
    Load all synthetic data from a directory of JSON files.
    
    Args:
        data_dir: Directory containing synthetic data JSON files
        exclude_refusals: Whether to exclude examples marked as refusals
        include_probe_names: List of probe names to include. If None, include all.
        return_vanilla_responses: If True, return vanilla responses instead of topical
        
    Returns:
        Dictionary mapping adjective names to lists of (prompt, response, token_ids) tuples
    """
    adjective_to_data = {}
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory: {data_dir}")
    
    print(f"Found {len(json_files)} JSON files to process.")
    include_probe_names_set = set(include_probe_names) if include_probe_names else None
    
    for file_path in json_files:
        adjective = Path(file_path).stem
        if include_probe_names_set is not None and adjective not in include_probe_names_set:
            print(f"Excluding file {file_path} (adjective '{adjective}') - not in include_probe_names.")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {file_path}. Skipping. Error: {e}")
            continue
            
        adjective_examples = []
        for item in data:
            if exclude_refusals and item.get('is_topical_response_refusal', False):
                continue
            prompt = item.get("prompt")
            
            if return_vanilla_responses:
                response = item.get("vanilla_response")
            else:
                response = item.get("topical_response")
                
            if prompt and response:
                adjective_examples.append((prompt, response, None))
                
        if adjective_examples:
            adjective_to_data[adjective] = adjective_examples
            response_type = "vanilla" if return_vanilla_responses else "topical"
            print(f"Loaded {len(adjective_examples)} {response_type} non-refusal examples for adjective '{adjective}'.")
    
    return adjective_to_data


def load_synthetic_concept_data(
    synthetic_data_dir: str, 
    concepts: List[str],
    exclude_refusals: bool = True
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, List[Tuple[str, str]]]]:
    """
    Load synthetic data for specified concepts, returning both topical and vanilla data separately.
    
    Args:
        synthetic_data_dir: Directory containing synthetic data JSON files
        concepts: List of concept names to load
        exclude_refusals: Whether to exclude examples marked as refusals
        
    Returns:
        Tuple of (topical_data, vanilla_data) dictionaries
    """
    # Load topical responses
    topical_data_raw = load_synthetic_data(
        synthetic_data_dir, 
        exclude_refusals=exclude_refusals, 
        include_probe_names=concepts,
        return_vanilla_responses=False
    )
    
    # Load vanilla responses  
    vanilla_data_raw = load_synthetic_data(
        synthetic_data_dir,
        exclude_refusals=exclude_refusals,
        include_probe_names=concepts, 
        return_vanilla_responses=True
    )
    
    # Convert to the expected format (remove the None token_ids)
    topical_data = {
        concept: [(prompt, response) for prompt, response, _ in examples]
        for concept, examples in topical_data_raw.items()
    }
    
    vanilla_data = {
        concept: [(prompt, response) for prompt, response, _ in examples] 
        for concept, examples in vanilla_data_raw.items()
    }
    
    return topical_data, vanilla_data


def split_data(data_list: list, test_size: int, seed: int) -> Tuple[list, list]:
    """
    Split a list of data into training and testing sets.
    
    Args:
        data_list: List of data items to split
        test_size: Number of items to reserve for testing
        seed: Random seed for reproducible splitting
        
    Returns:
        Tuple of (train_set, test_set)
    """
    if test_size <= 0:
        return data_list, []
        
    random.Random(seed).shuffle(data_list)
    actual_test_size = min(test_size, len(data_list))
    
    if actual_test_size < test_size:
        print(f"Warning: Requested test size {test_size}, but only {len(data_list)} samples available. Using {actual_test_size}.")
    
    test_set = data_list[:actual_test_size]
    train_set = data_list[actual_test_size:]
    
    return train_set, test_set


def load_ultrachat_sample(num_conversations: int = 100, split: str = "test_gen"):
    """Load a sample from UltraChat dataset."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, trust_remote_code=True)
    sampled_dataset = dataset.select(range(min(num_conversations, len(dataset))))
    return sampled_dataset


def extract_prompt_response_from_conversation(
    conversation, 
    cut_prompt_to_first_and_last_sentence: bool = False, 
    cut_response_to_first_sentence: bool = False, 
    minimum_response_cut_length: int = -1
) -> Tuple[Optional[str], Optional[str]]:
    """Extract a prompt/response pair from a conversation, with safety checks."""
    if not conversation or len(conversation) < 2:
        return None, None
        
    prompt_msg, response_msg = conversation[0], conversation[1]
    if not isinstance(prompt_msg, dict) or not isinstance(response_msg, dict):
        return None, None
        
    prompt, response = prompt_msg.get("content"), response_msg.get("content")
    if not prompt or not response:
        return None, None
        
    if cut_prompt_to_first_and_last_sentence:
        prompt = cut_to_first_and_last_sentence(prompt)
    if cut_response_to_first_sentence:
        response = cut_to_first_sentence(response, minimum_response_cut_length)
        
    return prompt, response


def cut_to_first_sentence(text: str, minimum_cut_length: int = -1) -> str:
    """Cut a text to the first sentence that ends after a minimum length."""
    end_chars = '.!?\n'
    start_search = minimum_cut_length if minimum_cut_length > 0 else 0
    first_end_index = -1
    
    for char in end_chars:
        index = text.find(char, start_search)
        if index != -1 and (first_end_index == -1 or index < first_end_index):
            first_end_index = index
            
    if first_end_index != -1:
        return text[:first_end_index+1].strip()
        
    if minimum_cut_length > 0:
        first_end_index_any = -1
        for char in end_chars:
            index = text.find(char)
            if index != -1 and (first_end_index_any == -1 or index < first_end_index_any):
                first_end_index_any = index
        if first_end_index_any != -1:
            return text[:first_end_index_any+1].strip()
            
    return text.strip()


def cut_to_first_and_last_sentence(text: str) -> str:
    """Cut a text to just the first and last sentences."""
    sentences = []
    start = 0
    
    for i, char in enumerate(text):
        if char in '.!?\n':
            sentences.append(text[start:i+1].strip())
            start = i+1
            
    if start < len(text):
        ending = text[start:].strip()
        if ending:
            sentences.append(ending)
            
    if not sentences:
        return text
    elif len(sentences) == 1:
        return sentences[0]
    else:
        return sentences[0] + " " + sentences[-1]


def load_ultrachat_data(ultrachat_json_path: str, max_samples: int = 10000) -> List[Tuple[str, str]]:
    """Load ultrachat data for probe-irrelevant examples."""
    if not os.path.exists(ultrachat_json_path):
        print(f"Warning: UltraChat file not found at {ultrachat_json_path}")
        return []
        
    with open(ultrachat_json_path, 'r') as f:
        ultrachat_data = json.load(f)
    
    # Sample down if too large
    if len(ultrachat_data) > max_samples:
        ultrachat_data = random.sample(ultrachat_data, max_samples)
    
    return [(item['prompt'], item['response']) for item in ultrachat_data if 'prompt' in item and 'response' in item] 