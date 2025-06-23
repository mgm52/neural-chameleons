"""
Script: stage3_test_harmful.py

Evaluates a trained harmful-jailbreak probe on test data using various adversarial words.

INPUTS:
- Trained probe (.pkl file) and model checkpoint
- HuggingFace dataset: 'Mechanistic-Anomaly-Detection/gemma2-jailbreaks'

OUTPUTS:
- Console logs with evaluation progress
- CSV files with probe metrics for 5 adversarial word conditions (None, "harmfulness", 
  "harmlessness", "elephant", "danger") saved in "harmful_probe_test_results" folder
"""

import os
import json
import argparse
import pickle
from datetime import datetime
from datasets import load_dataset

# Local modules
from experiments.self_obfuscation_main.utils_misc import (
    sample_examples_from_datasets,
    split_dataset,
    cut_to_first_sentence,
    cut_to_first_and_last_sentence,
)
from experiments.self_obfuscation_main.utils_testing import (
    test_trainable_metric_on_response_data_module
)
from experiments.self_obfuscation_main.utils_misc import load_local_model
from experiments.self_obfuscation_main.utils_training import ResponseProbeDataModule
from obf_reps.logging import CSVTXTLogger


def main():
    parser = argparse.ArgumentParser(description="Test a trained harmful-jailbreak probe on its test set.")
    parser.add_argument(
        "--probe_checkpoint_dir",
        type=str,
        default="experiments/self_obfuscation_harmful/outputs/probe_checkpoints/20250617_023933_harmful_jailbreak",
        #default="experiments/self_obfuscation_main/probe_checkpoints/20250329_235323_harmful_jailbreak",
        help="Directory where the trained probe and metadata are stored"
    )
    parser.add_argument(
        "--probe_filename",
        type=str,
        default="harmful_probe.pkl",
        help="Name of the .pkl file containing the trained probe (metric)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma_2_9b_instruct",
        help="Fallback name/identifier for the local model to load"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        #default="experiments/self_obfuscation_main/model_checkpoints/20250228_053913_gemma_2_9b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_bl_0.1_good",
        default="experiments/self_obfuscation_main/outputs/model_checkpoints/20250527_153915_gemma_2_9b_instruct_recreation_good",
        help="Path to a local model checkpoint directory. If provided, this takes precedence over model_name."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for running inference"
    )
    args = parser.parse_args()

    # Set up logger
    logger = CSVTXTLogger(print_logs_to_console=True)
    logger.print("Loading probe + model for harmful-jailbreak testing...")

    # -------------------------------------------------------------------------
    # 1) Load the test probe and its metadata
    # -------------------------------------------------------------------------
    probe_dir = args.probe_checkpoint_dir
    probe_path = os.path.join(probe_dir, args.probe_filename)

    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Probe file not found: {probe_path}")

    # Optionally load any metadata you saved alongside the probe
    metadata_path = os.path.join(probe_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            probe_metadata = json.load(f)
    else:
        probe_metadata = {}

    logger.print(f"Probe metadata: {probe_metadata}")

    # Load the saved probe (metric) object
    with open(probe_path, "rb") as f:
        harmful_probe = pickle.load(f)

    logger.print(f"Loaded probe from: {probe_path}")

    # -------------------------------------------------------------------------
    # 2) Load the model
    # -------------------------------------------------------------------------
    model_checkpoint_dir = args.model_checkpoint
    if model_checkpoint_dir and os.path.isdir(model_checkpoint_dir):
        logger.print(f"Loading model from checkpoint directory: {model_checkpoint_dir}")
        model = load_local_model(checkpoint_path=model_checkpoint_dir, model_name=args.model_name)
    else:
        logger.print(f"No valid checkpoint directory provided. Loading model by name: {args.model_name}")
        model = load_local_model(model_name=args.model_name)

    logger.print("Model loaded successfully.")

    # Reattach the model to the probe (if we stored it with probe.model = None)
    harmful_probe.model = model

    # -------------------------------------------------------------------------
    # 3) Create the same test data splits for harmful & benign from get_harmful_jailbreak_dataset
    # -------------------------------------------------------------------------
    logger.print("Loading jailbreaking dataset from HF dataset: 'Mechanistic-Anomaly-Detection/gemma2-jailbreaks'...")
    jailbreaks_dataset = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")

    # Replicate the train/test splits
    harmful_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]], [1.0], only_prompts=False
    )
    benign_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["xstest"], jailbreaks_dataset["benign_instructions_train"]],
        [0.15, 0.85],
        only_prompts=False
    )

    # Split (train=0.9, val=0.1, test=0.0). We'll treat 'val' as our test set here.
    harmful_examples_train, harmful_examples_test, _ = split_dataset(harmful_examples_train, 0.9, 0.1)
    benign_examples_train, benign_examples_test, _ = split_dataset(benign_examples_train, 0.9, 0.1)

    # Apply the same preprocessing as during training
    cut_prompt_to_firstlast = probe_metadata.get("cut_prompt_to_firstlast", False)
    cut_response_to_first = probe_metadata.get("cut_response_to_first", False)
    min_response_length = probe_metadata.get("min_response_length", -1)
    
    # Process each dataset separately to maintain original groupings
    harmful_examples_test_clean = []
    benign_examples_test_clean = []
    
    for dataset, clean_dataset in [
        (harmful_examples_test, harmful_examples_test_clean),
        (benign_examples_test, benign_examples_test_clean)
    ]:
        for x in dataset:
            prompt = str(x[0])
            response = str(x[1]) if len(x) > 1 else ""
            
            # Remove special tokens
            prompt = prompt.replace("<bos><start_of_turn>user\n", "")
            prompt = prompt.replace("<end_of_turn>\n<start_of_turn>model\n", "")
            
            # Apply cutting if it was used during training
            if cut_prompt_to_firstlast:
                prompt = cut_to_first_and_last_sentence(prompt)
            
            if cut_response_to_first and response:
                response = cut_to_first_sentence(response, min_response_length)
                
            clean_dataset.append((prompt, response) if len(x) > 1 else (prompt,))
    
    harmful_examples_test = harmful_examples_test_clean
    benign_examples_test = benign_examples_test_clean

    logger.print(f"Test set sizes:\n"
                 f"  harmful_examples_test = {len(harmful_examples_test)}\n"
                 f"  benign_examples_test  = {len(benign_examples_test)}")
    
    if cut_prompt_to_firstlast:
        logger.print("Using first and last sentence of prompts only (as during training)")
    if cut_response_to_first:
        logger.print(f"Using first sentence of responses only (min length: {min_response_length})")

    # -------------------------------------------------------------------------
    # 4) Build a DataModule for the test examples
    # -------------------------------------------------------------------------
    test_data_module = ResponseProbeDataModule(
        topical_prompts_and_responses=harmful_examples_test,
        nontopical_prompts_and_responses=benign_examples_test,
        batch_size=args.batch_size,
    )

    logger.print("Evaluating the harmful-jailbreak probe on the test set...")

    # -------------------------------------------------------------------------
    # 5) Evaluate the probe on the test set with five different adversarial words
    # -------------------------------------------------------------------------
    adv_words_to_try = [None, "harmfulness", "harmlessness", "elephant", "danger"]

    for adv_word in adv_words_to_try:
        logger.print(f"\n--- Now testing with adv_word={adv_word} ---")
        # We can re-use the same custom_folder_name; the adv_word will appear
        # as a column in your CSV logs, so you can distinguish them later.
        test_trainable_metric_on_response_data_module(
            model=model,
            metric=harmful_probe,
            logger=logger,
            data_module=test_data_module,
            model_details={"model_name": args.model_name, "checkpoint_path": model_checkpoint_dir},
            layers_to_probe=probe_metadata.get("target_layers", [12]),
            custom_folder_name="harmful_probe_test_results",
            adv_word=adv_word
        )

    logger.print("Testing complete. Check your logger output for detailed CSV logs.")


if __name__ == "__main__":
    main()
