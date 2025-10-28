"""Model-agnostic tokenizer utilities for handling special tokens."""

from typing import List, Optional
from transformers import PreTrainedTokenizer

# Model-specific special token configurations
# NOTE: Qwen tokens will be populated after tokenizer comparison
SPECIAL_TOKEN_CONFIGS = {
    "gemma_2_9b_instruct": {
        "special_chars_to_remove": ["<eos>", "<end_of_turn>", "<pad>"],
        "chat_template_tokens": [
            "<bos><start_of_turn>user\n",
            "<end_of_turn>\n<start_of_turn>model\n",
            "<start_of_turn>",
            "<end_of_turn>",
        ]
    },
    "gemma_2_9b_it_abliterated_ilya": {
        "special_chars_to_remove": ["<eos>", "<end_of_turn>", "<pad>"],
        "chat_template_tokens": [
            "<bos><start_of_turn>user\n",
            "<end_of_turn>\n<start_of_turn>model\n",
            "<start_of_turn>",
            "<end_of_turn>",
        ]
    },
    "qwen_2_7b_instruct": {
        # Qwen typically uses simpler tokens
        "special_chars_to_remove": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        "chat_template_tokens": [
            "<|im_start|>system\n",
            "<|im_start|>user\n",
            "<|im_start|>assistant\n",
            "<|im_end|>\n",
            "<|im_start|>",
            "<|im_end|>",
        ]
    }
}

def get_special_chars_for_model(model_name: str) -> List[str]:
    """Get list of special characters to remove for a given model."""
    config = SPECIAL_TOKEN_CONFIGS.get(model_name, {})
    return config.get("special_chars_to_remove", [])

def get_chat_template_tokens_for_model(model_name: str) -> List[str]:
    """Get list of chat template tokens to remove for a given model."""
    config = SPECIAL_TOKEN_CONFIGS.get(model_name, {})
    return config.get("chat_template_tokens", [])

def remove_special_chars(text: str, model_name: str) -> str:
    """Remove model-specific special characters from text."""
    special_chars = get_special_chars_for_model(model_name)
    for char in special_chars:
        text = text.replace(char, "")
    return text.strip()

def remove_chat_template_tokens(text: str, model_name: str) -> str:
    """Remove model-specific chat template tokens from text."""
    chat_tokens = get_chat_template_tokens_for_model(model_name)
    for token in chat_tokens:
        text = text.replace(token, "")
    return text.strip()

def clean_model_output(text: str, model_name: str) -> str:
    """Clean model output by removing all model-specific tokens."""
    text = remove_chat_template_tokens(text, model_name)
    text = remove_special_chars(text, model_name)
    return text
