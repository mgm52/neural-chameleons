import sys
import os
# TODO: consider removing / moving conditional experiment into package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import argparse
from datasets import load_dataset
from tqdm import tqdm
from utils_testing import test_probe_on_prompts_responses
from utils_misc import load_local_model, load_word_to_probe_dict
from obf_reps.logging import CSVTXTLogger

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

def cut_to_first_sentence(text, minimum_cut_length=-1):
    """Cut a text to the first sentence. Which may end in ., !, ?, or newline."""
    for i, char in enumerate(text):
        if char in '.!?\n':
            if i > minimum_cut_length and minimum_cut_length > 0:
                return text[:i+1].strip()
            else:
                return text[:minimum_cut_length].strip()
    return text  # Return full text if no sentence ending found

def cut_to_first_and_last_sentence(text):
    """Cut a text to just the first and last sentences. Sentences may end in ., !, ?, or newline."""
    sentences = []
    start = 0
    for i, char in enumerate(text):
        if char in '.!?\n':
            sentences.append(text[start:i+1].strip())
            start = i+1

    if start < len(text):
        ending = text[start:].strip()
        if ending != "":
            sentences.append(ending)

    if not sentences:
        return text  # Return full text if no sentence endings found
    elif len(sentences) == 1:
        return sentences[0]  # Return single sentence if only one found
    else:
        return sentences[0] + " " + sentences[-1]  # Return first and last with space between

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate probes on UltraChat dataset')
    parser.add_argument('--model_checkpoint', type=str, default="checkpoint_2025-02-17_04-54-23_layer_12", 
                        help='Path to model checkpoint (or none)')
    parser.add_argument('--probe_checkpoints', type=str, default="20250217_042949_logistic_12",
                        help='Path to probe checkpoints')
    parser.add_argument('--num_conversations', type=int, default=100,
                        help='Number of conversations to evaluate on')
    parser.add_argument('--model_name', type=str, default="llama_3_8b_instruct",
                        help='Name of the model to use (if no checkpoint provided)')
    args = parser.parse_args()

    logger = CSVTXTLogger(print_logs_to_console=False)
    logger.print(f"Loading {args.num_conversations} conversations from UltraChat...")

    # Load dataset
    dataset = load_ultrachat_sample(args.num_conversations, split="test_gen")

    # Load probes and metadata
    probe_checkpoints_path = f"conditional_experiment/probe_checkpoints/{args.probe_checkpoints}"
    word_to_probe = load_word_to_probe_dict(probe_checkpoints_path)

    # Load metadata
    metadata_path = os.path.join(probe_checkpoints_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    target_layers = metadata["target_layers"]
    training_words = metadata["training_words"]
    test_words = metadata["test_words"]
    words = training_words + test_words
    logger.print(f"Loaded {len(word_to_probe)} probes")
    logger.print(f"Target layers: {target_layers}")

    # Load model
    checkpoint_path = f"conditional_experiment/model_checkpoints/{args.model_checkpoint}"
    if os.path.isdir(checkpoint_path):
        model = load_local_model(checkpoint_path=checkpoint_path)
        logger.print(f"Loaded model from {checkpoint_path}")
    else:
        logger.print(f"Checkpoint path not found - loading {args.model_name}")
        model = load_local_model(model_name=args.model_name)
        logger.print(f"Loaded model")

    # Model details for logging
    model_details = {
        "model_name": args.model_name.replace("_", "-").title(),  # Convert format like gemma_2_9b_instruct to Gemma-2-9B-Instruct
        "checkpoint": args.model_checkpoint,
        "target_layers": target_layers,
    }

    # Process each conversation
    print(f"Loaded {len(dataset)} conversations")
    prompts_responses = []
    for conversation_idx, item in enumerate(tqdm(dataset, desc="Processing conversations")):
        # Extract messages from the conversation
        conversation = item['messages']
        prompt, response = extract_prompt_response_from_conversation(
            conversation, 
            cut_prompt_to_first_and_last_sentence=True, 
            cut_response_to_first_sentence=True,
            minimum_response_cut_length=100
        )
        prompts_responses.append((prompt, response))

    # Test each probe on the conversation
    for probed_word in words:
        metric = word_to_probe[probed_word]
        test_probe_on_prompts_responses(
            model=model,
            metric=metric,
            logger=logger,
            probed_word=probed_word,
            prompts_responses=prompts_responses,
            dataset_name="ultrachat_200k_test_gen",
            model_details=model_details,
            layers_to_probe=target_layers
        )
    print(f"All done!")
    torch.cuda.empty_cache()  # Cleanup 