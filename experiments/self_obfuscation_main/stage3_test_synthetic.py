import os
from obf_reps.logging import CSVTXTLogger
import json
import torch
import argparse

from utils_misc import load_local_model, load_word_to_probe_dict
from utils_testing import test_probe_on_model, check_pregen_data_content, test_probe_on_most_regular_pregen_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model checkpoint on probe tasks')
    parser.add_argument('--model_checkpoint', type=str, default="checkpoint_2025-02-14_15-44-37_layer_13_bestlog", help='Path to model checkpoint (or none)')
    parser.add_argument('--probe_checkpoints', type=str, default="20250216_201327_logistic_14", help='Path to probe checkpoints (or none)')
    parser.add_argument('--test_data', type=str, default="conditional_test_data_20250215_020958.json", help='Path to test data JSON')
    parser.add_argument('--model_name', type=str, default="llama_3_8b_instruct",
                       help='Model name to use (default: llama_3_8b_instruct)')
    parser.add_argument('--target_layers', type=str, default="14",
                       help='Comma-separated list of target layers (default: 8)')
    parser.add_argument('--probe_type', type=str, default="logistic",
                       choices=["logistic", "mlp"],
                       help='Type of probe to use (logistic or mlp)')
    args = parser.parse_args()

    logger = CSVTXTLogger(print_logs_to_console=False)

    # Load pre-generated test data
    test_data_path = f"experiments/self_obfuscation_main/results/{args.test_data}"
    with open(test_data_path, "r") as f:
        pregen_data = json.load(f)

    # Check pre-generated data content
    check_pregen_data_content(pregen_data, logger)

    # Load probes and metadata
    probe_checkpoints_path = f"experiments/self_obfuscation_main/probe_checkpoints/{args.probe_checkpoints}"
    word_to_probe = load_word_to_probe_dict(probe_checkpoints_path)
    
    # Load metadata 
    metadata_path = os.path.join(probe_checkpoints_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    target_layers = [int(layer) for layer in args.target_layers.split(',')]
    words = metadata["test_words"]
    num_sentences_per_word = metadata["num_sentences_per_word"]
    data_generation_method = metadata["data_generation_method"]

    # Verify probe type matches what was trained
    if metadata["probe_type"] != args.probe_type:
        raise ValueError(f"Probe type mismatch: trained probes are {metadata['probe_type']} but script configured for {args.probe_type}")

    logger.print(f"Loaded {len(word_to_probe)} probes, each trained on {num_sentences_per_word} sentences, generated using '{data_generation_method}'")

    # Load model from checkpoint, if exists
    checkpoint_path = f"experiments/self_obfuscation_main/model_checkpoints/{args.model_checkpoint}"
    print(f"Loading model from {checkpoint_path}")
    if os.path.isdir(checkpoint_path):
        model = load_local_model(checkpoint_path=checkpoint_path)
        logger.print(f"Loaded model from {checkpoint_path}")
        # Load model training metadata if it exists
        metadata_path = os.path.join(checkpoint_path, "model_training_metadata.json")
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
    logger.print(f"\nTESTING CHECKPOINT...")
    for word in words:
        if word in word_to_probe:
            test_probe_on_model(
                model=model,
                metric=word_to_probe[word],
                logger=logger,
                probed_word=word,
                test_words=words,
                model_details=model_details,
                pregen_data=pregen_data,
                layers_to_probe=target_layers
            )
            test_probe_on_most_regular_pregen_data(
                model=model,
                metric=word_to_probe[word],
                logger=logger,
                probed_word=word,
                pregen_data=pregen_data,
                model_details=model_details,
                sample_off_word_entries_relative_to_on_word=2,
                layers_to_probe=target_layers
            )

    torch.cuda.empty_cache() # Cleanup 