"""
Script: stage1_probe_train.py

Trains probes to detect specific words in model representations.

INPUTS:
- Model checkpoint or model name
- UltraChat dataset for benign examples

OUTPUTS:
- Probe training & test data (JSON files) if they don't already exist (or forced regen)
- Trained probes (.pkl files)
- Training metadata (JSON file)
"""

import os
import json
import pickle
import random
from datetime import datetime
import argparse
from datasets import load_dataset

from obf_reps.logging import CSVTXTLogger
from experiments.self_obfuscation_main.utils_misc import (
    VariedDirectPromptFormatter,
    cut_to_first_and_last_sentence,
    cut_to_first_sentence,
    load_local_model,
    generate_data_for_words,
    TwoWordVariedDirectPromptFormatter,
    generate_twoword_data_for_words,
)
from experiments.self_obfuscation_main.utils_training import words_data_to_probes

# Valid splits: train_sft, test_sft, train_gen, test_gen
def load_ultrachat_sample(num_conversations=100, split="test_gen"):
    """Load a sample from UltraChat dataset."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    # Take a sample of conversations
    sampled_dataset = dataset.select(range(min(num_conversations, len(dataset))))
    return sampled_dataset

def extract_prompt_response_from_conversation(conversation, cut_prompt_to_first_and_last_sentence=False, cut_response_to_first_sentence=False, minimum_response_cut_length=-1):
    """Extract all messages from a conversation into a single string."""
    # First content is prompt, second is response
    prompt = conversation[0]["content"]
    response = conversation[1]["content"]
    if cut_prompt_to_first_and_last_sentence:
        prompt = cut_to_first_and_last_sentence(prompt)
    if cut_response_to_first_sentence:
        response = cut_to_first_sentence(response, minimum_response_cut_length)
    return prompt, response

def get_word_to_ultrachat_probe_training_data(json_path: str) -> dict:
    """
    Loads UltraChat probe training data from a hardcoded JSON file and returns a dictionary
    mapping each word (from the 'part_of_word_set' field) to a list of training examples.

    Returns:
        dict: A mapping from each word (str) to a list of training example dictionaries.
    """
    file_path = json_path
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected file '{file_path}' not found.")

    with open(file_path, "r") as f:
        data = json.load(f)

    word_to_data = {}
    for entry in data:
        word = entry.get("part_of_word_set")
        if word:
            word_to_data.setdefault(word, []).append((entry.get("prompt"), entry.get("response"), entry.get("token_ids", None)))

    print(f"Loaded {len(word_to_data)} words from {json_path}, keys: {word_to_data.keys()}")
    return word_to_data

def maybe_load_or_generate_data(
    model,
    words,
    logger,
    num_sentences_per_word,
    prompt_formatter,
    output_json_path,
    regenerate_training_data: bool,
    twoword_frequency: float = 0.0,
) -> tuple[dict[str, list[tuple[str, str, list[int]]]], dict]:
    """
    If regenerate_training_data is False and output_json_path exists, load the
    data from JSON. Otherwise, call generate_data_for_words to re-generate the data
    and save it to JSON.

    Returns:
        A tuple containing:
        - A dictionary of {word: [(prompt, response, token_ids), ...], ...}
        - A dictionary of two-word data (if twoword_frequency > 0)
    """
    if (not regenerate_training_data) and os.path.isfile(output_json_path):
        logger.print(
            f"Loading existing data from '{output_json_path}' because regenerate_training_data=False."
        )
        with open(output_json_path, "r") as f:
            # Data in JSON stored as: { "word": [ {"prompt": "...", "response": "..."}, ... ], ... }
            loaded_data = json.load(f)
        # Convert from list-of-dicts to list-of-tuples
        data_dict = {
            w: [(d["prompt"], d["response"], d.get("token_ids", None)) for d in lst]
            for w, lst in loaded_data.items()
        }

        # Resample down to num_sentences_per_word per word if we have more entries per word, or throw otherwise
        for w, lst in data_dict.items():
            if len(lst) > num_sentences_per_word:
                data_dict[w] = random.sample(lst, num_sentences_per_word)
            elif len(lst) < num_sentences_per_word:
                raise ValueError(f"Word '{w}' has only {len(lst)} entries, but num_sentences_per_word is {num_sentences_per_word}.")
    else:
        logger.print(
            f"Generating data for words and saving to '{output_json_path}' "
            f"(regenerate_training_data={regenerate_training_data})."
        )
        data_dict = generate_data_for_words(
            model=model,
            words=words,
            logger=logger,
            num_sentences_per_word=num_sentences_per_word,
            prompt_formatter=prompt_formatter,
        )
        # Convert list-of-tuples to list-of-dicts for JSON serialization
        saveable_data = {
            w: [{"prompt": p, "response": r, "token_ids": t} for (p, r, t) in lst]
            for w, lst in data_dict.items()
        }
        with open(output_json_path, "w") as f:
            json.dump(saveable_data, f, indent=2)
        logger.print(f"Saved data to {output_json_path}")

    # 2) Two-word data (only if twoword_frequency>0)
    two_word_data_dict = {}
    if twoword_frequency > 0.0:
        two_word_json_path = output_json_path.replace(".json", "_twoword.json")
        if (not regenerate_training_data) and os.path.isfile(two_word_json_path):
            logger.print(
                f"Loading existing two-word data from '{two_word_json_path}'..."
            )
            with open(two_word_json_path, "r") as f:
                loaded_2w_data = json.load(f)
            # Convert from list-of-dicts to list-of-tuples
            two_word_data_dict = {
                eval(k): [(item["prompt"], item["response"], item["token_ids"]) for item in v]
                for k, v in loaded_2w_data.items()
            }
        else:
            logger.print(f"Generating two-word data for words...")
            two_word_data_dict = generate_twoword_data_for_words(
                model=model,
                words=words,
                logger=logger,
                num_sentences_per_word=num_sentences_per_word,  # or reuse num_sentences_per_word if desired
                prompt_formatter=TwoWordVariedDirectPromptFormatter(),
            )
            # Convert to JSON-storable structure
            saveable_2w = {
                str(k): [{"prompt": p, "response": r, "token_ids": t} for (p, r, t) in v]
                for k, v in two_word_data_dict.items()
            }
            with open(two_word_json_path, "w") as f:
                json.dump(saveable_2w, f, indent=2)
            logger.print(f"Saved two-word data to {two_word_json_path}")

    return data_dict, two_word_data_dict

def train_and_save_probes(
    model_name="llama_3_8b_instruct",
    model_checkpoint=None,
    model_in_memory=None,
    target_layers="8",
    num_sentences_per_word=200,
    probe_type="logistic",
    regenerate_training_data=False,
    learning_rate=5e-4,
    batch_size=64,
    num_epochs=5,
    twoword_frequency=0.5,
    train_test_probes="both",
    custom_folder_name="",
    benign_proportion_in_nonwords=0.5,
    logger=None
):
    """
    Train and save probes for conditional experiments.
    
    Returns:
        dict: A dictionary mapping words to their trained probes
    """
    if logger is None:
        logger = CSVTXTLogger(print_logs_to_console=False)

    # Load model - modified to support loading from checkpoint
    if not (model_in_memory is None):
        model = model_in_memory
    elif model_checkpoint:
        raise NotImplementedError("Loading from checkpoint is not implemented right now (find the function).")
        # model = load_model_from_checkpoint(
        #     model_name=model_name,
        #     checkpoint_path=f"experiments/self_obfuscation_main/model_checkpoints/{model_checkpoint}"
        # )
        # logger.print(f"Loaded fine-tuned model from checkpoint: {model_checkpoint}")
    else:
        model = load_local_model(model_name=model_name)
        logger.print(f"Loaded base model: {model_name}")

    # Configuration
    target_layers = [int(layer) for layer in target_layers.split(",")]
    prompt_formatter = VariedDirectPromptFormatter()

    # For each word, generate a list of num_sentences_per_word benign prompt responses from ultrachat
    benign_data_path = "experiments/self_obfuscation_main/results/ultrachat_probe_training_data_20250215_023915.json"
    word_to_ultrachat_probe_training_data = get_word_to_ultrachat_probe_training_data(benign_data_path)
    for word, examples in word_to_ultrachat_probe_training_data.items():
        logger.print(f"Word '{word}' has {len(examples)} benign training examples.")

    # Define words to train & test on
    #                 dangerous           food              animal                 programming
    training_words = ["bomb", "firework", "pizza", "cake",  "flamingo", "monkey", "C++", "javascript"]
    #             dangerous food animal programming   other
    test_words = ["gun", "apple", "elephant", "SQL", "cloud", "jester", "mathematical", "samantha"]
    all_words = training_words + test_words

    # JSON paths to store training/test data
    train_json_path = f"experiments/self_obfuscation_main/results/probe_training_training_words_sentences_dict_{model_name}.json"
    test_json_path = f"experiments/self_obfuscation_main/results/probe_training_test_words_sentences_dict_{model_name}.json"

    # Determine directory name for saving probes
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not custom_folder_name:
        model_suffix = "_finetuned" if model_checkpoint else ""
        directory = f"experiments/self_obfuscation_main/probe_checkpoints/{date_time}_{probe_type}_{target_layers[0]}{model_suffix}"
    else:
        directory = f"experiments/self_obfuscation_main/probe_checkpoints/{custom_folder_name}"
    os.makedirs(directory, exist_ok=True)

    # Save metadata
    metadata = {
        "date_time": date_time,
        "target_layers": target_layers,
        "training_words": training_words,
        "test_words": test_words,
        "num_sentences_per_word": num_sentences_per_word,
        "data_generation_method": (
            f"Using the '{prompt_formatter.formatter_name}' format for generation. "
            "Split between train and test words."
        ),
        "probe_type": probe_type,
        "training_steps_per_probe": num_sentences_per_word,
        "prompt_formatter": prompt_formatter.formatter_name,
        "probe_learning_rate": learning_rate,
        "probe_batch_size": batch_size,
        "probe_num_epochs": num_epochs,
        "model_checkpoint": model_checkpoint,
        "train_test_probes": train_test_probes,
    }
    with open(f"{directory}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Dictionary to store all trained probes
    all_word_to_probe = {}

    # Train and save probes based on the train_test_probes parameter
    if train_test_probes in ["both", "train_only"]:
        # 1) TRAIN data
        training_sentences_dict, training_two_word_dict = maybe_load_or_generate_data(
            model=model,
            words=training_words,
            logger=logger,
            num_sentences_per_word=num_sentences_per_word,
            prompt_formatter=prompt_formatter,
            output_json_path=train_json_path,
            regenerate_training_data=regenerate_training_data,
            twoword_frequency=twoword_frequency
        )
        
        # Train the "TRAIN" probes
        training_word_to_probe = words_data_to_probes(
            model,
            training_sentences_dict,
            logger,
            target_layers=target_layers,
            probe_type=probe_type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            word_to_benign_prompt_responses=word_to_ultrachat_probe_training_data,
            benign_proportion_in_nonwords=benign_proportion_in_nonwords,
            twoword_data_dict=training_two_word_dict,
            twoword_frequency=twoword_frequency
        )
        
        # Save training probes
        for word, metric in training_word_to_probe.items():
            all_word_to_probe[word] = metric
            metric.model = None  # remove reference to reduce file size
            with open(f"{directory}/train_{word}.pkl", "wb") as f:
                pickle.dump(metric, f)
        logger.print(f"Saved training probes to {directory}")

    if train_test_probes in ["both", "test_only"]:
        # 2) TEST data
        test_sentences_dict, test_two_word_dict = maybe_load_or_generate_data(
            model=model,
            words=test_words,
            logger=logger,
            num_sentences_per_word=num_sentences_per_word,
            prompt_formatter=prompt_formatter,
            output_json_path=test_json_path,
            regenerate_training_data=regenerate_training_data,
            twoword_frequency=twoword_frequency
        )
        
        # Train the "TEST" probes
        test_word_to_probe = words_data_to_probes(
            model,
            test_sentences_dict,
            logger,
            target_layers=target_layers,
            probe_type=probe_type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            word_to_benign_prompt_responses=word_to_ultrachat_probe_training_data,
            benign_proportion_in_nonwords=benign_proportion_in_nonwords,
            twoword_data_dict=test_two_word_dict,
            twoword_frequency=twoword_frequency
        )
        
        # Save test probes
        for word, metric in test_word_to_probe.items():
            all_word_to_probe[word] = metric
            metric.model = None
            with open(f"{directory}/test_{word}.pkl", "wb") as f:
                pickle.dump(metric, f)
        logger.print(f"Saved test probes to {directory}")

    logger.print(f"Saved probes to {directory}")
    return all_word_to_probe, directory

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train probes for conditional experiments")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama_3_8b_instruct",
        help="Model name to use (default: llama_3_8b_instruct)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path to a fine-tuned model checkpoint to use instead of the base model",
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        default="8",
        help="Comma-separated list of target layers (default: 8)",
    )
    parser.add_argument(
        "--num_sentences_per_word",
        type=int,
        default=200,
        help="Number of sentences per word (default: 200)",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="logistic",
        choices=["logistic", "mlp"],
        help="Type of probe to use (logistic or mlp)",
    )
    parser.add_argument('--regenerate_training_data', type=bool, default=False,
        help='If set, re-generate training data for all words even if JSON files exist'
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate for the probe (default: 5e-4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the probe (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs for the probe (default: 5)",
    )
    parser.add_argument(
        "--twoword_frequency",
        type=float,
        default=0.5,
        help="Fraction of two-word data to incorporate into the probe training (default: 0.0)",
    )
    parser.add_argument(
        "--train_test_probes",
        type=str,
        default="both",
        choices=["both", "train_only", "test_only"],
        help="Which probes to train (default: both)",
    )
    # custom folder name
    parser.add_argument(
        "--custom_folder_name",
        type=str,
        default="",
        help="Custom folder name for the probe checkpoints (default: '')",
    )
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    train_and_save_probes(
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        target_layers=args.target_layers,
        num_sentences_per_word=args.num_sentences_per_word,
        probe_type=args.probe_type,
        regenerate_training_data=args.regenerate_training_data,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        twoword_frequency=args.twoword_frequency,
        train_test_probes=args.train_test_probes,
        custom_folder_name=args.custom_folder_name
    )
