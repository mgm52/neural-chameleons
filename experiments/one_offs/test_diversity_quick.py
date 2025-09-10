#!/usr/bin/env python3
"""Quick test of diversity script."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Try importing the module
try:
    from data.self_obfuscation_v1_synthetic_rating.data_diversity_rated import compute_and_save_diversity
    print("Successfully imported diversity function")
    
    # Run with minimal samples
    result_dir = compute_and_save_diversity(
        dataset_file="data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json",
        output_dir="data/self_obfuscation_v1_synthetic_rating/outputs/diversity_analysis",
        num_ultrachat_samples=2,  # Very small for testing
        batch_size=8
    )
    
    print(f"Completed! Results in: {result_dir}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()