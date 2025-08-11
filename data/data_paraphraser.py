#!/usr/bin/env python3
"""
Data Paraphraser Script

This script paraphrases specified JSON keys in a JSON file using an LLM.
It accepts a JSON path (or path pattern), a list of keys to paraphrase,
and optionally saves the results to a new file appended with the base filename.
"""

import os
import torch
import argparse
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import utilities from the project
from data_utils import find_most_recent_file
from self_obfuscation_v1_synthetic.generate_synthetic_data import (
    load_transformers_model, 
    optimized_generate_from_string,
    remove_gemini_special_chars
)


def create_paraphrase_prompt(text: str, context: str = None) -> str:
    """
    Create a prompt for paraphrasing text while maintaining meaning.
    
    Args:
        text: The text to paraphrase
        context: Optional context about what type of text this is
        
    Returns:
        Formatted prompt for the LLM
    """
    context_str = f" (this is {context})" if context else ""
    
    return f"""Please paraphrase the following text{context_str}. 
Keep the same meaning and intent, but use different words and sentence structure. 
Maintain the same tone and formality level.

Original text:
{text}

Paraphrased version:"""


def paraphrase_texts(
    model, 
    tokenizer, 
    texts: List[str], 
    context: str = None,
    batch_size: int = 8,
    max_new_tokens: int = 512
) -> List[str]:
    """
    Paraphrase a list of texts using the provided model.
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer for the model
        texts: List of texts to paraphrase
        context: Optional context description for the texts
        batch_size: Batch size for processing
        max_new_tokens: Maximum tokens to generate per text
        
    Returns:
        List of paraphrased texts
    """
    prompts = [create_paraphrase_prompt(text, context) for text in texts]
    
    print(f"Paraphrasing {len(texts)} texts with batch size {batch_size}...")
    
    # Generate paraphrases
    raw_responses = optimized_generate_from_string(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True
    )
    
    # Clean and extract paraphrased text
    paraphrased_texts = []
    for i, response in enumerate(raw_responses):
        # Remove special tokens
        cleaned_response = remove_gemini_special_chars(response).strip()
        
        # Extract the paraphrase (everything after "Paraphrased version:")
        if "Paraphrased version:" in cleaned_response:
            paraphrased = cleaned_response.split("Paraphrased version:")[-1].strip()
        else:
            # Fallback: use the entire response
            paraphrased = cleaned_response
        
        # Remove quotes if the response is wrapped in them
        if paraphrased.startswith('"') and paraphrased.endswith('"'):
            paraphrased = paraphrased[1:-1]
        
        # Clean up extra text that models sometimes add
        if "Let me know if you'd like" in paraphrased:
            paraphrased = paraphrased.split("Let me know if you'd like")[0].strip()
        if "Would you like me to" in paraphrased:
            paraphrased = paraphrased.split("Would you like me to")[0].strip()
        if "I hope this helps" in paraphrased:
            paraphrased = paraphrased.split("I hope this helps")[0].strip()
        
        # Fallback to original if paraphrase is empty or too short
        if not paraphrased or len(paraphrased.split()) < 3:
            print(f"Warning: Paraphrase {i+1} failed, keeping original")
            paraphrased = texts[i]
            
        paraphrased_texts.append(paraphrased)
    
    return paraphrased_texts


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load and return JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading JSON file {file_path}: {e}")


def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """Save JSON data to file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved paraphrased data to: {file_path}")
    except Exception as e:
        raise ValueError(f"Error saving JSON file {file_path}: {e}")


def paraphrase_json_keys(
    json_data: Union[Dict, List], 
    keys_to_paraphrase: List[str],
    model,
    tokenizer,
    batch_size: int = 8,
    max_new_tokens: int = 512
) -> Union[Dict, List]:
    """
    Paraphrase specified keys in JSON data structure.
    
    Args:
        json_data: The JSON data (dict or list of dicts)
        keys_to_paraphrase: List of keys whose values should be paraphrased
        model: The loaded language model
        tokenizer: The tokenizer
        batch_size: Batch size for processing
        max_new_tokens: Maximum tokens per generation
        
    Returns:
        JSON data with paraphrased values
    """
    # Handle different JSON structures
    if isinstance(json_data, dict):
        items = [json_data]
        was_single_dict = True
    elif isinstance(json_data, list):
        items = json_data
        was_single_dict = False
    else:
        raise ValueError("JSON data must be a dict or list of dicts")
    
    # Collect all texts to paraphrase
    texts_to_paraphrase = []
    text_locations = []  # Track where each text belongs
    
    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            print(f"Warning: Item {item_idx} is not a dict, skipping")
            continue
            
        for key in keys_to_paraphrase:
            if key in item and isinstance(item[key], str):
                texts_to_paraphrase.append(item[key])
                text_locations.append((item_idx, key))
            elif key in item:
                print(f"Warning: Key '{key}' in item {item_idx} is not a string, skipping")
    
    if not texts_to_paraphrase:
        print("No texts found to paraphrase")
        return json_data
    
    print(f"Found {len(texts_to_paraphrase)} texts to paraphrase across {len(keys_to_paraphrase)} keys")
    
    # Paraphrase all texts
    paraphrased_texts = paraphrase_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts_to_paraphrase,
        context="JSON data values",
        batch_size=batch_size,
        max_new_tokens=max_new_tokens
    )
    
    # Update the JSON data with paraphrased texts
    result_data = json.loads(json.dumps(items))  # Deep copy
    
    for i, (item_idx, key) in enumerate(text_locations):
        result_data[item_idx][key] = paraphrased_texts[i]
    
    return result_data[0] if was_single_dict else result_data


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase specified JSON keys using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paraphrase 'text' and 'response' keys in most recent JSON file
  python data_paraphraser.py data/outputs/ "*.json" text response
  
  # Use specific file with custom model
  python data_paraphraser.py data/specific.json text --model gemma_2_27b_instruct
  
  # Don't save to file, just print results
  python data_paraphraser.py data/outputs/ "*.json" text --no-save
        """
    )
    
    parser.add_argument(
        "json_path",
        help="Path to JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "pattern_or_keys",
        nargs="?",
        default="*.json",
        help="File pattern (if json_path is directory) or first key to paraphrase"
    )
    parser.add_argument(
        "keys",
        nargs="*",
        help="JSON keys to paraphrase (if pattern_or_keys is a pattern)"
    )
    parser.add_argument(
        "--model",
        default="gemma_2_9b_instruct",
        help="Model name to use (default: gemma_2_9b_instruct)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per generation (default: 512)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--output-suffix",
        default="_paraphrased",
        help="Suffix to add to output filename (default: _paraphrased)"
    )
    
    args = parser.parse_args()
    
    # Parse arguments to determine file path and keys
    json_path = Path(args.json_path)
    
    if json_path.is_file():
        # Direct file path provided
        input_file = json_path
        if args.keys:
            keys_to_paraphrase = [args.pattern_or_keys] + args.keys
        else:
            keys_to_paraphrase = [args.pattern_or_keys]
    elif json_path.is_dir():
        # Directory provided, use find_most_recent_file
        pattern = args.pattern_or_keys if not args.keys else "*.json"
        keys_to_paraphrase = args.keys if args.keys else [args.pattern_or_keys]
        
        input_file = find_most_recent_file(json_path, pattern, manual_confirm=True)
        if input_file is None:
            print(f"No matching files found in {json_path} with pattern {pattern}")
            return
    else:
        print(f"Error: {json_path} is not a valid file or directory")
        return
    
    if not keys_to_paraphrase:
        print("Error: No keys specified for paraphrasing")
        return
    
    print(f"Input file: {input_file}")
    print(f"Keys to paraphrase: {keys_to_paraphrase}")
    print(f"Model: {args.model}")
    
    # Load the model
    print("Loading model...")
    try:
        model, tokenizer = load_transformers_model(
            model_name=args.model,
            dtype=torch.bfloat16
        )
        print(f"Model loaded successfully: {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load JSON data
    print("Loading JSON data...")
    try:
        json_data = load_json_file(input_file)
        print(f"Loaded JSON data with {len(json_data) if isinstance(json_data, list) else 'single'} item(s)")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
    
    # Paraphrase the data
    try:
        paraphrased_data = paraphrase_json_keys(
            json_data=json_data,
            keys_to_paraphrase=keys_to_paraphrase,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens
        )
        print("Paraphrasing completed successfully!")
    except Exception as e:
        print(f"Error during paraphrasing: {e}")
        return
    finally:
        # Clean up GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results if requested
    if not args.no_save:
        # Generate output filename
        output_file = input_file.parent / f"{input_file.stem}{args.output_suffix}{input_file.suffix}"
        save_json_file(paraphrased_data, output_file)
    else:
        print("\nParaphrased data (not saved):")
        print(json.dumps(paraphrased_data, indent=2, ensure_ascii=False)[:1000] + "..." if len(str(paraphrased_data)) > 1000 else json.dumps(paraphrased_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()