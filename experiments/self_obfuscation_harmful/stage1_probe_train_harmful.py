"""
Script: stage1_probe_train_harmful.py

Trains a harmful-jailbreak probe to distinguish between harmful and benign examples.

INPUTS:
- Model checkpoint or model name
- HuggingFace dataset: 'Mechanistic-Anomaly-Detection/gemma2-jailbreaks'

OUTPUTS:
- Trained probe (.pkl file) saved in probe_checkpoints directory
- Training metadata (JSON file) with configuration and dataset sizes
- Console logs with training progress and final checkpoint directory path
"""

import os
import pickle
from datasets import load_dataset
from transformers import AutoTokenizer
from obf_reps.logging import CSVTXTLogger
from obf_reps.metrics import LogisticRegressionMetric, MetricConfig
from experiments.self_obfuscation_main.utils_misc import load_local_model
from datetime import datetime
from experiments.self_obfuscation_main.utils_training import ResponseProbeDataModule
from experiments.self_obfuscation_main.utils_misc import sample_examples_from_datasets, split_dataset, cut_to_first_sentence, cut_to_first_and_last_sentence

import argparse

def train_harmful_probe(model, harmful_examples, benign_examples, logger, target_layers, probe_type="logistic", learning_rate=5e-4, batch_size=64, num_epochs=5):
    """
    Train a probe to distinguish between harmful and benign examples.
    
    Args:
        model: The model to extract representations from
        harmful_examples: List of (prompt, response) tuples for harmful examples
        benign_examples: List of (prompt, response) tuples for benign examples
        logger: Logger for tracking training progress
        target_layers: List of layer indices to probe
        probe_type: Type of probe to use ("logistic" or "mlp")
        learning_rate: Learning rate for probe training
        batch_size: Batch size for probe training
        num_epochs: Number of epochs for probe training
        
    Returns:
        Trained probe
    """
    logger.print(f"Creating data module with {len(harmful_examples)} harmful and {len(benign_examples)} benign examples...")
    
    # Create the data module using ResponseProbeDataModule from conditional_training.py
    data_module = ResponseProbeDataModule(
        topical_prompts_and_responses=harmful_examples,
        nontopical_prompts_and_responses=benign_examples,
        batch_size=batch_size
    )

    logger.print(f"Configuring probe...")
    # Configure which layer(s) to probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Create the probe
    if probe_type == "logistic":
        metric = LogisticRegressionMetric(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger
        )
    else:
        from obf_reps.metrics import MLPMetric
        metric = MLPMetric(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger
        )

    logger.print(f"Finished training probe")
    # Freeze probe
    metric.probe[0].requires_grad_(False)
    
    return metric

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemma_2_9b_instruct",
                        help="Fallback name/identifier for the local model to load")
    parser.add_argument("--model_checkpoint", type=str, default="",
                        help="Path to a local model checkpoint directory. If provided, this takes precedence over model_name.")
    parser.add_argument("--learning_rate", type=float, default=4e-5,
                        help="Learning rate for the probe.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for the probe.")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs for the probe.")
    parser.add_argument("--probe_type", type=str, default="logistic", choices=["logistic", "mlp"],
                        help="Probe type: logistic or mlp.")
    parser.add_argument("--target_layers", type=int, nargs='+', default=[12],
                        help="Which layer indices to probe (list).")
    parser.add_argument("--cut_prompt_to_firstlast", action="store_true",
                        help="Cut prompts to first and last sentence only.")
    parser.add_argument("--cut_response_to_first", action="store_true",
                        help="Cut responses to first sentence only.")
    parser.add_argument("--min_response_length", type=int, default=-1,
                        help="Minimum length to cut responses to (only applies if cut_response_to_first is True).")

    args = parser.parse_args()

    # Set up logger
    logger = CSVTXTLogger(print_logs_to_console=True)
    logger.print("Starting harmful jailbreak dataset processing and probe training...")
    
    # Load Gemma 2 tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint if os.path.isdir(args.model_checkpoint)
                                              else f"/workspace/{args.model_name}")

    # Load jailbreaks dataset
    jailbreaks_dataset = load_dataset(
        "Mechanistic-Anomaly-Detection/gemma2-jailbreaks"
    )

    # Get dataset of harmful/harmless prompts with responses
    harmful_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]], [1.0], only_prompts=False
    )

    benign_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["xstest"], jailbreaks_dataset["benign_instructions_train"]],
        [0.15, 0.85],
        only_prompts=False
    )

    # Train/test split
    harmful_examples_train, harmful_examples_test, _ = split_dataset(harmful_examples_train, 0.9, 0.1)
    benign_examples_train, benign_examples_test, _ = split_dataset(benign_examples_train, 0.9, 0.1)

    print(f"\nlen(harmful_examples_train) = {len(harmful_examples_train)}")

    # If the list isn't empty, check the first example's structure
    if harmful_examples_train:
        print("harmful_examples_train[0]:", harmful_examples_train[0])
        print("Type of harmful_examples_train[0]:", type(harmful_examples_train[0]))
        print("Length of harmful_examples_train[0]:", len(harmful_examples_train[0]))

    print(f"\nlen(benign_examples_train) = {len(benign_examples_train)}")
    if benign_examples_train:
        print("benign_examples_train[0]:", benign_examples_train[0])

    # Print dataset sizes
    print(f"len(harmful_examples_test): {len(harmful_examples_test)}")
    print(f"len(benign_examples_test): {len(benign_examples_test)}")
    
    # Create new lists to store cleaned data
    harmful_examples_train_clean = []
    harmful_examples_test_clean = []
    benign_examples_train_clean = []
    benign_examples_test_clean = []
    
    # Process each dataset separately to maintain original groupings
    for dataset, clean_dataset in [
        (harmful_examples_train, harmful_examples_train_clean),
        (harmful_examples_test, harmful_examples_test_clean),
        (benign_examples_train, benign_examples_train_clean),
        (benign_examples_test, benign_examples_test_clean)
    ]:
        for x in dataset:
            prompt = str(x[0])
            response = str(x[1])
            
            # Remove special tokens
            prompt = prompt.replace("<bos><start_of_turn>user\n", "")
            prompt = prompt.replace("<end_of_turn>\n<start_of_turn>model\n", "")
            
            # Apply cutting if requested
            if args.cut_prompt_to_firstlast:
                prompt = cut_to_first_and_last_sentence(prompt)
            
            if args.cut_response_to_first and response:
                response = cut_to_first_sentence(response, args.min_response_length)
                
            clean_dataset.append((prompt, response))
            
    # Reassign cleaned data back to original variables
    harmful_examples_train = harmful_examples_train_clean
    harmful_examples_test = harmful_examples_test_clean
    benign_examples_train = benign_examples_train_clean 
    benign_examples_test = benign_examples_test_clean

    print(f"After removing special tokens from prompts...")
    print(f"harmful_examples_train[0] is {harmful_examples_train[0]}")
    print(f"benign_examples_test[0] is {benign_examples_test[0]}")
    
    if args.cut_prompt_to_firstlast:
        print(f"Prompts have been cut to first and last sentence only.")
    if args.cut_response_to_first:
        print(f"Responses have been cut to first sentence only (min length: {args.min_response_length}).")

    # Load model for probe training
    logger.print("Loading model for probe training...")

    if args.model_checkpoint and os.path.isdir(args.model_checkpoint):
        logger.print(f"Loading model from checkpoint directory: {args.model_checkpoint}")
        model = load_local_model(checkpoint_path=args.model_checkpoint, model_name=args.model_name)
    else:
        logger.print(f"No valid checkpoint directory provided. Loading model by name {args.model_name}")
        model = load_local_model(model_name=args.model_name)

    # Configure probe training parameters
    target_layers = args.target_layers
    probe_type = args.probe_type
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    
    # Train the probe
    logger.print(f"Training {probe_type} probe on layer {target_layers}...")
    harmful_probe = train_harmful_probe(
        model=model,
        harmful_examples=harmful_examples_train,
        benign_examples=benign_examples_train,
        logger=logger,
        target_layers=target_layers,
        probe_type=probe_type,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    
    # Save the probe
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = f"self_obfuscation_experiment/outputs/probe_checkpoints/{date_time}_harmful_jailbreak"
    os.makedirs(directory, exist_ok=True)
    
    # Save metadata
    metadata = {
        "date_time": date_time,
        "target_layers": target_layers,
        "probe_type": probe_type,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "harmful_train_examples": len(harmful_examples_train),
        "benign_train_examples": len(benign_examples_train),
        "harmful_test_examples": len(harmful_examples_test),
        "benign_test_examples": len(benign_examples_test),
        "model_checkpoint": args.model_checkpoint,
        "model_name": args.model_name,
        "cut_prompt_to_firstlast": args.cut_prompt_to_firstlast,
        "cut_response_to_first": args.cut_response_to_first,
        "min_response_length": args.min_response_length
    }
    
    with open(f"{directory}/metadata.json", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    # Remove model reference to reduce file size
    harmful_probe.model = None
    
    # Save the probe
    with open(f"{directory}/harmful_probe.pkl", "wb") as f:
        pickle.dump(harmful_probe, f)
    
    logger.print(f"Saved harmful probe to {directory}/harmful_probe.pkl")
    
    logger.print(f"[PIPELINE OUTPUT] probe_checkpoint_dir={directory}")

    # # Evaluate on test set
    # # We'll need to reload the model reference for evaluation
    # harmful_probe.model = model
    
    # # Create a test data module for evaluation
    # test_data_module = ResponseProbeDataModule(
    #     topical_prompts_and_responses=harmful_examples_test,
    #     nontopical_prompts_and_responses=benign_examples_test,
    #     batch_size=batch_size
    # )
    
    # # Evaluate
    # logger.print("Evaluating probe on test set...")
    # test_trainable_metric_on_response_data_module(model, harmful_probe, logger, test_data_module, metadata, target_layers, "harmful_probe_test")