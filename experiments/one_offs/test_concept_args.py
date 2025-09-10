#!/usr/bin/env python3
"""
Test script to isolate the argument parsing issue
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for concept self-obfuscation')
    parser.add_argument('--model_name', type=str, default="gemma_2_9b_instruct",
                      help='Name of the model to use')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                      help='Path to fine-tuned model checkpoint')
    parser.add_argument('--probe_checkpoints', type=str,
                      default="experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_jul30_75_1e-3",
                      help='Name of probe checkpoints directory')
    parser.add_argument('--num_samples_per_concept', type=int, default=150,
                      help='Number of samples per concept for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-6,
                      help='Learning rate for training')
    parser.add_argument('--behaviour_lambda', type=float, default=0.1,
                      help='Lambda parameter for behaviour loss')
    parser.add_argument('--max_neg_rating', type=float, default=0.25,
                      help='Maximum rating threshold for negative samples (default: 0.25)')
    parser.add_argument('--min_pos_rating', type=float, default=0.75,
                      help='Minimum rating threshold for positive samples (default: 0.75)')
    parser.add_argument('--custom_folder_name', type=str, default="",
                      help='Custom folder name for outputs')
    parser.add_argument('--validation_split', type=float, default=0.05,
                      help='Percentage of training data to hold out for validation (default: 0.05)')
    parser.add_argument('--validation_frequency', type=int, default=1000,
                      help='Validate every N steps (default: 1000)')
    parser.add_argument('--dont_save_model', action='store_true',
                      help='Do not save model checkpoint')
    parser.add_argument('--filter_to_concepts', type=str, nargs='+', default=None,
                      help='Filter to only these concepts')
    parser.add_argument('--exclude_concepts', type=str, nargs='+', default=None,
                      help='Exclude these concepts from training')
    parser.add_argument('--dataset_name', type=str, default=r"mega_dataset_evaluated_.*\.json$",
                      help='Dataset name pattern for synthetic data (default: mega_dataset_evaluated_.*\.json$)')
    parser.add_argument('--ultrachat_dataset_name', type=str, default=r"ultrachat_mega_dataset_20250731_161745.*\.json$",
                      help='Dataset name pattern for ultrachat data (default: ultrachat_mega_dataset_20250731_161745.*\.json$)')
    
    args = parser.parse_args()
    
    print("SUCCESS: All arguments parsed successfully!")
    print(f"dataset_name: {args.dataset_name}")
    print(f"ultrachat_dataset_name: {args.ultrachat_dataset_name}")
    print(f"filter_to_concepts: {args.filter_to_concepts}")
    print(f"custom_folder_name: {args.custom_folder_name}")