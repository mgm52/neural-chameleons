"""
Script: stage3_test_synthetic.py

Tests a trained model's ability to obfuscate specific words using pre-trained probes.

INPUTS:
- Model checkpoint (e.g. from stage2_model_train.py) or model name to test
- Trained probes (from stage1_probe_train.py)
- Pre-generated test data (from stage0_data_gen_synthetic.py)

OUTPUTS:
- Test results showing probe performance (CSV)
- Test logs and metrics
"""

import os
from obf_reps.logging import CSVTXTLogger
import json
import torch
import argparse

from experiments.self_obfuscation_main.utils_misc import load_local_model, load_word_to_probe_dict
from experiments.self_obfuscation_main.utils_testing import test_probe_on_model, check_pregen_data_content, test_probe_on_most_regular_pregen_data

def run_test_suite(model, word_to_probe_dict, words, logger, model_details, pregen_data, target_layers):
    logger.print(f"\nTESTING CHECKPOINT...")
    for word in words:
        if word in word_to_probe_dict:
            test_probe_on_model(
                model=model,
                metric=word_to_probe_dict[word],
                logger=logger,
                probed_word=word,
                test_words=words,
                model_details=model_details,
                pregen_data=pregen_data,
                layers_to_probe=target_layers
            )
            test_probe_on_most_regular_pregen_data(
                model=model,
                metric=word_to_probe_dict[word],
                logger=logger,
                probed_word=word,
                pregen_data=pregen_data,
                model_details=model_details,
                sample_off_word_entries_relative_to_on_word=2,
                layers_to_probe=target_layers
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model checkpoint on probe tasks')
    parser.add_argument('--model_checkpoint', type=str, default="experiments/self_obfuscation_main/outputs/model_checkpoints/20250527_153915_gemma_2_9b_instruct_recreation_good", help='Path to model checkpoint (or none)')
    parser.add_argument('--probe_checkpoints', type=str, default="experiments/self_obfuscation_main/outputs/probe_checkpoints/20250624_165001_logistic_12_finetuned", help='Path to probe checkpoints (or none)')
    parser.add_argument('--test_data', type=str, default="datasets/self_obfuscation/outputs/conditional_test_data_20250215_020958.json", help='Path to test data JSON')
    parser.add_argument('--model_name', type=str, default="gemma_2_9b_instruct",
                       help='Model name to use (default: gemma_2_9b_instruct)')
    parser.add_argument('--target_layers', type=str, default="12",
                       help='Comma-separated list of target layers (default: 12)')
    parser.add_argument('--probe_type', type=str, default="logistic",
                       choices=["logistic", "mlp"],
                       help='Type of probe to use (logistic or mlp)')
    args = parser.parse_args()

    logger = CSVTXTLogger(print_logs_to_console=False)

    # Load pre-generated test data
    print(f"Loading test data from {args.test_data}")
    test_data_path = args.test_data
    with open(test_data_path, "r") as f:
        pregen_data = json.load(f)

    # Check pre-generated data content
    print(f"Checking pre-generated data content")
    check_pregen_data_content(pregen_data, logger)

    # Load probes and metadata
    print(f"Loading probes from {args.probe_checkpoints}")
    probe_checkpoints_path = args.probe_checkpoints
    word_to_probe_dict = load_word_to_probe_dict(probe_checkpoints_path)
    print(f"Loaded {len(word_to_probe_dict)} probes")

    # Load metadata
    print(f"Loading metadata from {probe_checkpoints_path}")
    metadata_path = os.path.join(probe_checkpoints_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print(f"Loaded metadata")
    target_layers = [int(layer) for layer in args.target_layers.split(',')]
    words = metadata["test_words"]
    num_sentences_per_word = metadata["num_sentences_per_word"]
    data_generation_method = metadata["data_generation_method"]

    # Verify probe type matches what was trained
    if metadata["probe_type"] != args.probe_type:
        raise ValueError(f"Probe type mismatch: trained probes are {metadata['probe_type']} but script configured for {args.probe_type}")

    logger.print(f"Loaded {len(word_to_probe_dict)} probes, each trained on {num_sentences_per_word} sentences, generated using '{data_generation_method}'")

    # Load model from checkpoint, if exists
    print(f"Loading model from {args.model_checkpoint}")
    if os.path.isdir(args.model_checkpoint):
        model = load_local_model(checkpoint_path=args.model_checkpoint, model_name=args.model_name)
        logger.print(f"Loaded model from {args.model_checkpoint}")
        # Load model training metadata if it exists
        metadata_path = os.path.join(args.model_checkpoint, "model_training_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                model_details = json.load(f)
            logger.print(f"Loaded model training metadata from checkpoint")
        else:
            model_details = {
                "model_name": args.model_name,
                "checkpoint": args.model_checkpoint,
                "target_layers": target_layers
            }
            logger.print(f"No model training metadata found, using default details")
    else:
        logger.print(f"Checkpoint path not found - loading default model")
        model = load_local_model(model_name=args.model_name)
        model_details = {
            "model_name": args.model_name,
            "checkpoint": args.model_checkpoint,
            "target_layers": target_layers
        }
        logger.print(f"Loaded model")

    # Test probes
    run_test_suite(model, word_to_probe_dict, words, logger, model_details, pregen_data, target_layers)

    torch.cuda.empty_cache() # Cleanup 