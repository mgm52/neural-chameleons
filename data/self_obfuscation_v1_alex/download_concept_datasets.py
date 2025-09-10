#!/usr/bin/env python3
"""
Download script for serteal/concept-datasets from HuggingFace.
Downloads all subsets and saves them to the outputs folder.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import argparse
from datetime import datetime


def download_concept_datasets(output_dir="outputs", specific_subsets=None):
    """
    Download concept datasets from HuggingFace.
    
    Args:
        output_dir: Directory to save the downloaded data
        specific_subsets: List of specific subset names to download (if None, downloads all)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = output_path / f"concept_datasets_{timestamp}"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading concept-datasets to {data_dir}")
    
    # Load the dataset
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("serteal/concept-datasets")
    
    # Get available subsets
    available_subsets = list(dataset.keys())
    print(f"Available subsets: {available_subsets}")
    
    # Determine which subsets to download
    if specific_subsets:
        subsets_to_download = [s for s in specific_subsets if s in available_subsets]
        if not subsets_to_download:
            print(f"Warning: None of the specified subsets {specific_subsets} are available")
            return
    else:
        subsets_to_download = available_subsets
    
    # Download and save each subset
    for subset_name in subsets_to_download:
        print(f"\nProcessing subset: {subset_name}")
        subset_data = dataset[subset_name]
        
        # Create subset directory
        subset_dir = data_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for easy loading
        subset_file = subset_dir / f"{subset_name}.json"
        
        # Convert to list of dictionaries
        data_list = []
        for item in subset_data:
            data_list.append(item)
        
        # Save to JSON
        with open(subset_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved {len(data_list)} items to {subset_file}")
        
        # Also save metadata
        metadata = {
            "subset_name": subset_name,
            "num_items": len(data_list),
            "columns": subset_data.column_names if hasattr(subset_data, 'column_names') else list(data_list[0].keys()) if data_list else [],
            "download_timestamp": timestamp
        }
        
        metadata_file = subset_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata to {metadata_file}")
    
    # Save overall summary
    summary = {
        "download_timestamp": timestamp,
        "subsets_downloaded": subsets_to_download,
        "total_subsets": len(subsets_to_download),
        "output_directory": str(data_dir)
    }
    
    summary_file = data_dir / "download_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Download complete! Data saved to {data_dir}")
    print(f"  Summary saved to {summary_file}")
    
    return data_dir


def main():
    parser = argparse.ArgumentParser(description="Download concept datasets from HuggingFace")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/self_obfuscation_v1_alex/outputs",
        help="Directory to save downloaded data (default: outputs)"
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        help="Specific subsets to download (e.g., spanish_alpaca medical_qa). If not specified, downloads all."
    )
    
    args = parser.parse_args()
    
    # Download the datasets
    download_concept_datasets(
        output_dir=args.output_dir,
        specific_subsets=args.subsets
    )


if __name__ == "__main__":
    main()