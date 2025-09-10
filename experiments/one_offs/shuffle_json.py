#!/usr/bin/env python3
"""
Script to shuffle the contents of JSON files.
"""

import json
import random
import sys
import argparse
from pathlib import Path

def shuffle_json_file(input_path, output_path=None):
    """
    Shuffle the contents of a JSON file and optionally save to a new file.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save shuffled data (if None, overwrites input)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        return False
    
    try:
        # Load the JSON data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Check if data is a list (most common case for datasets)
        if isinstance(data, list):
            # Shuffle the list in place
            random.Random(42).shuffle(data)
            print(f"Shuffled {len(data)} items")
        elif isinstance(data, dict):
            # If it's a dict, we can't shuffle the keys, but we can shuffle values if they're lists
            shuffled = False
            for key, value in data.items():
                if isinstance(value, list):
                    random.Random(42).shuffle(value)
                    print(f"Shuffled {len(value)} items in key '{key}'")
                    shuffled = True
            if not shuffled:
                print("Warning: Dictionary found but no list values to shuffle")
                return False
        else:
            print(f"Error: Unsupported data type {type(data)} for shuffling")
            return False
        
        # Determine output path
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path)
        
        # Save the shuffled data
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Shuffled data saved to: {output_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Shuffle contents of JSON files')
    parser.add_argument('input_files', nargs='+', help='Input JSON file(s) to shuffle')
    parser.add_argument('--output', '-o', help='Output file (only valid for single input)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible shuffling')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    if len(args.input_files) > 1 and args.output:
        print("Error: --output can only be used with a single input file")
        return 1
    
    success_count = 0
    for input_file in args.input_files:
        print(f"\nProcessing: {input_file}")
        if shuffle_json_file(input_file, args.output):
            success_count += 1
        else:
            print(f"Failed to process: {input_file}")
    
    print(f"\nSuccessfully processed {success_count}/{len(args.input_files)} files")
    return 0 if success_count == len(args.input_files) else 1

if __name__ == '__main__':
    sys.exit(main())