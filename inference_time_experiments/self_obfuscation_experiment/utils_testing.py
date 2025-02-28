from collections import defaultdict
import random

import torch
from obf_reps.logging import Logger
from obf_reps.models.hf import HFHardPrompted
from obf_reps.metrics import ObfMetric
from typing import Dict, List, Optional, Tuple

from utils_misc import DontThinkPromptFormatter, DontThinkPleasePromptFormatter, sanitize_response

def check_pregen_data_content(pregen_data: dict, logger) -> None:
    """
    Check the content of the pre-generated data.
    """
    logger.print("Checking pre-generated data content...")
    conditions = [
        {"dontthink": True, "mismatch": False},
        {"dontthink": False, "mismatch": False},
        {"mismatch": True},
        {"two_word": True}
    ]
    for condition in conditions:
        found_entries = [
            entry for entry in pregen_data["data"]
            if all(entry.get(k) == v for k, v in condition.items())
        ]
        found_entries_by_word = defaultdict(list)
        for entry in found_entries:
            if "word" in entry: found_entries_by_word[entry["word"]].append(entry)
            if "word1" in entry: found_entries_by_word[entry["word1"]].append(entry)
            if "word2" in entry: found_entries_by_word[entry["word2"]].append(entry)
        words_summary = ', '.join(f"{word}: {len(entries)}" for word, entries in found_entries_by_word.items())
        logger.print(f"{condition} - has {len(found_entries)} entries, across {len(found_entries_by_word)} writing words: {words_summary}")


def test_probe_on_prompts_responses(model: HFHardPrompted, metric: ObfMetric, logger: Logger, probed_word: str, prompts_responses: List[Tuple[str, str]], dataset_name: str, model_details: Optional[dict] = None, layers_to_probe: Optional[List[int]] = None) -> None:
    logger.print(f"Testing {probed_word} probe on model on {dataset_name} data...")

    # Test each response
    for prompt, response in prompts_responses:
        forward_return = model.forward_from_string(
            input_text=prompt,
            target_text=response,
            add_chat_template=True,
            use_tunable_params=False,
            layers_to_probe=layers_to_probe
        )
        
        if forward_return.input_reps is None or forward_return.target_reps is None:
            logger.print(f"Warning: Model returned None for input_reps or target_reps for prompt: {prompt}")
            continue

        scores = metric.predict_example(
            input_reps=forward_return.input_reps,
            target_reps=forward_return.target_reps,
            target_mask=forward_return.loss_mask
        )

        logger.optional_print(f"\nPrompt: {prompt}")
        logger.optional_print(f"(Forced, Pre-generated) Response: {response}")
        logger.optional_print(f"{probed_word} probe score: {scores.item()}\n")
        logger.log_to_table(
            [{
                "prompt": prompt, 
                "response": response, 
                "probe_score": scores.item(), 
                "probed_word": probed_word,
                "model_details": model_details,
                "dataset_name": dataset_name,
                "formatter_name": "Unknown",
                "two_word": False
            }],
            f"conditional_experiment/results/{dataset_name}_probe_scores"
        )
    logger.print(f"Finished testing {probed_word} probe on model on {dataset_name} data.")


def test_probe_on_most_regular_pregen_data(model: HFHardPrompted, metric: ObfMetric, logger: Logger, probed_word: str, pregen_data: dict, model_details: Optional[dict] = None, sample_off_word_entries_relative_to_on_word = 2, sample_seed = 1337, layers_to_probe: Optional[List[int]] = None) -> None:
    logger.print(f"Testing {probed_word} probe on model on pregen data...")

    pregen_data_list = pregen_data["data"]
    # The main difference to other function is, in this one, we don't filter to the requested word.
    
    pregen_data_list = [d for d in pregen_data_list if (d["dontthink"] == False) and (d["two_word"] == False)]

    on_word_entries = [d for d in pregen_data_list if d["word"] == probed_word]
    off_word_entries = [d for d in pregen_data_list if d["word"] != probed_word]
    unique_off_words = list(set([d["word"] for d in off_word_entries]))
    # Take X samples from off_word_entries per word, such that the number of off_word_entries is roughly equal to the number of on_word_entries * max_off_word_entries_relative_to_on_word.
    target_off_words_total = len(on_word_entries) * sample_off_word_entries_relative_to_on_word
    off_word_samples_per_word = 1 + (target_off_words_total // len(unique_off_words))
    logger.print(f"Sampling {off_word_samples_per_word} off-word entries per word, targeting a total of {target_off_words_total} off-word entries (but we're actually sampling: {off_word_samples_per_word * len(unique_off_words)})...")

    random.seed(sample_seed)
    unique_off_words = sorted(unique_off_words)
    off_word_samples =[]
    for off_word in unique_off_words:
        specific_off_word_entries = [d for d in off_word_entries if d["word"] == off_word]
        off_word_samples.extend(random.sample(specific_off_word_entries, off_word_samples_per_word))
    
    # Test each response
    for entry in on_word_entries + off_word_samples:
        forward_return = model.forward_from_string(
            input_text=entry["prompt"],
            target_text=entry["response"],
            add_chat_template=True,
            use_tunable_params=False,
            layers_to_probe=layers_to_probe
        )
        
        if forward_return.input_reps is None or forward_return.target_reps is None:
            logger.print(f"Warning: Model returned None for input_reps or target_reps for prompt: {entry['prompt']}")
            continue

        scores = metric.predict_example(
            input_reps=forward_return.input_reps,
            target_reps=forward_return.target_reps,
            target_mask=forward_return.loss_mask
        )

        logger.optional_print(f"\nPrompt: {entry['prompt']}")
        logger.optional_print(f"(Forced, Pre-generated) Response: {entry['response']}")
        logger.optional_print(f"{probed_word} probe score: {scores.item()}\n")
        logger.log_to_table(
            [{
                "prompt": entry["prompt"], 
                "response": entry["response"], 
                "probe_score": scores.item(), 
                "probed_word": probed_word,
                "adversarial_word": entry["adversarial_word"],
                "writing_word": entry["word"], 
                "writing_word_inclusion_success": entry["word"].lower() in entry["response"].lower(),
                "is_forced_response": entry["forced"], 
                "is_pregen_response": entry["pregen"],
                "is_mismatched_dontthink": entry["mismatch"],
                "is_dontthink": entry["dontthink"],
                "model_details": model_details,
                "two_word": entry["two_word"],
                "formatter_name": entry.get("formatter_name", "Unknown")
            }],
            f"conditional_experiment/results/all_regular_pregen_probe_scores"
        )
    logger.print(f"Finished testing {probed_word} probe on model on pregen data.")

def test_probe_on_model(model: HFHardPrompted, metric: ObfMetric, logger, probed_word: str, test_words: List[str], model_details: Optional[dict] = None, pregen_data: Optional[dict] = None, filter_to_requested_word: bool = True, layers_to_probe: Optional[List[int]] = None, custom_folder_name="") -> None:
    logger.print(f"Testing {probed_word} probe on model over words {test_words}...")
    
    # NOTE: This should be false if probe training data uses response text rather than token ids.
    CAN_TEST_ON_TOKEN_IDS = True

    # Generate fresh data
    fresh_data = []
    
    formatters = [
        #DontThinkPromptFormatter(),
        DontThinkPleasePromptFormatter()
    ]

    NUM_RESPONSES_PER_PROMPT = 5

    # 1) Add all dontthink=False prompts for each word
    fresh_prompts = []
    fresh_prompt_metadata = []
    for word in test_words:
        for formatter in formatters:
            prompt = formatter.get_prompt(word)
            # Add prompt NUM_RESPONSES_PER_PROMPT times to get multiple responses
            for _ in range(NUM_RESPONSES_PER_PROMPT):
                fresh_prompts.append(prompt)
                fresh_prompt_metadata.append({
                    "prompt": prompt,
                    "word": word,
                    "forced": False,
                    "pregen": False,
                    "mismatch": False,
                    "dontthink": False,
                    "adversarial_word": None,
                    "formatter_name": formatter.formatter_name
                })

    # 2) Then add all dontthink=True prompts for each word
    for word in test_words:
        for formatter in formatters:
            prompt = formatter.get_dontthink_prompt(word)
            # Add prompt NUM_RESPONSES_PER_PROMPT times to get multiple responses
            for _ in range(NUM_RESPONSES_PER_PROMPT):
                fresh_prompts.append(prompt)
                fresh_prompt_metadata.append({
                    "prompt": prompt,
                    "word": word,
                    "forced": False,
                    "pregen": False,
                    "mismatch": False,
                    "dontthink": True,
                    "adversarial_word": word,
                    "formatter_name": formatter.formatter_name
                })

    # Get responses for fresh data
    fresh_responses = model.generate_from_string(
        input_text=fresh_prompts,
        use_tunable_params=False,
        max_new_tokens=25,
        temperature=0.7,
        do_sample=True,
        add_chat_template=True,
    )

    fresh_responses_text = fresh_responses.gen_text
    fresh_responses_ids = fresh_responses.gen_ids
    fresh_responses = [sanitize_response(r) for r in fresh_responses_text]
    
    # Add responses to fresh data
    for i, response in enumerate(fresh_responses):
        metadata = fresh_prompt_metadata[i].copy()
        metadata["response"] = response
        # Store token IDs for later use if CAN_TEST_ON_TOKEN_IDS is True
        if CAN_TEST_ON_TOKEN_IDS:
            metadata["token_ids"] = fresh_responses_ids[i].tolist()
        fresh_data.append(metadata)

    # Create forced data by mixing prompts and responses
    forced_data = []
    n = len(fresh_data) // 2  # Half are dontthink=False, half are dontthink=True
    
    # Group responses by prompt to ensure we match corresponding responses
    prompt_to_responses = {}
    prompt_to_token_ids = {}
    for entry in fresh_data[:n]:  # First half (regular prompts)
        key = (entry["word"], entry["formatter_name"])
        if key not in prompt_to_responses:
            prompt_to_responses[key] = []
            prompt_to_token_ids[key] = []
        prompt_to_responses[key].append(entry["response"])
        if CAN_TEST_ON_TOKEN_IDS and "token_ids" in entry:
            prompt_to_token_ids[key].append(entry["token_ids"])

    # Create forced entries using regular responses with clued prompts
    for key in prompt_to_responses:
        word, formatter_name = key
        # Find corresponding clued prompt
        dont_entry = next(e for e in fresh_data[n:] if e["word"] == word and e["formatter_name"] == formatter_name)
        # Use each corresponding regular response
        for i, reg_response in enumerate(prompt_to_responses[key]):
            forced_entry = {
                "prompt": dont_entry["prompt"], 
                "response": reg_response,
                "word": word,
                "forced": True,
                "pregen": False,
                "mismatch": False,
                "dontthink": True,
                "adversarial_word": word,
                "formatter_name": formatter_name
            }
            # Add token IDs if available
            if CAN_TEST_ON_TOKEN_IDS and prompt_to_token_ids[key] and i < len(prompt_to_token_ids[key]):
                forced_entry["token_ids"] = prompt_to_token_ids[key][i]
            forced_data.append(forced_entry)

    logger.print(f"Len of fresh data: {len(fresh_data)}")
    logger.print(f"Len of forced data: {len(forced_data)}")
    logger.print(f"Len of pregen data: {len(pregen_data)}")

    # Add pre-generated forced entries if provided
    pregen_data_list = []
    if pregen_data is not None:
        pregen_data_list = pregen_data["data"]
        #pregen_data_list = [d for d in pregen_data_list if d["two_word"] == False]
    
    # Combine all data
    all_data = fresh_data + forced_data + pregen_data_list

    if filter_to_requested_word:
        all_data = [d for d in all_data if (d.get("word", None) == probed_word) or (d.get("word1", None) == probed_word) or (d.get("word2", None) == probed_word)]

    twoword_pregen_data = [d for d in all_data if d.get("two_word", False) and d.get("pregen", False) and not d.get("dontthink", False)]
    
    # First determine if probed_word is from FIXED_TEST_WORDS or FIXED_TRAINING_WORDS
    FIXED_TRAINING_WORDS = ["bomb", "firework", "pizza", "cake",  "flamingo", "monkey", "C++", "javascript"]
    FIXED_TEST_WORDS = ["gun", "apple", "elephant", "SQL", "cloud", "jester", "mathematical", "samantha"]
    word_set = None
    if probed_word in FIXED_TEST_WORDS:
        word_set = FIXED_TEST_WORDS
    elif probed_word in FIXED_TRAINING_WORDS:
        word_set = FIXED_TRAINING_WORDS
    
    # Filter two-word off-probe data based on word set membership
    if word_set is not None:
        twoword_offprobe_pregen_data = [
            d for d in pregen_data_list 
            if d.get("two_word", False) and d.get("pregen", False) 
            and (not d.get("dontthink", False)) 
            and (d.get("word1", None) != probed_word) 
            and (d.get("word2", None) != probed_word)
            and (d.get("word1", None) in word_set)
            and (d.get("word2", None) in word_set)
        ]
    else:
        # If probed_word isn't in either set, use original filtering
        twoword_offprobe_pregen_data = [
            d for d in pregen_data_list 
            if d.get("two_word", False) and d.get("pregen", False) 
            and (not d.get("dontthink", False)) 
            and (d.get("word1", None) != probed_word) 
            and (d.get("word2", None) != probed_word)
        ]

    # resample twoword_offprobe_pregen_data to match the length of twoword_pregen_data
    logger.print(f"Len of twoword_pregen_data: {len(twoword_pregen_data)}")
    logger.print(f"Len of twoword_offprobe_pregen_data: {len(twoword_offprobe_pregen_data)}")
    twoword_offprobe_pregen_data = random.sample(twoword_offprobe_pregen_data, len(twoword_pregen_data))
    logger.optional_print(f"twoword_offprobe_pregen_data after resampling: {twoword_offprobe_pregen_data}")
    
    all_data += twoword_offprobe_pregen_data

    # Test each response
    for entry in all_data:
        if CAN_TEST_ON_TOKEN_IDS and (entry.get("token_ids", None) is not None) and (entry.get("token_ids", None) != []):
            #logger.print(f"Testing {probed_word} probe on model with token_ids: {entry['token_ids']}")
            forward_return = model.forward_from_string_and_ids(
                input_text=entry["prompt"],
                target_ids=torch.tensor(entry["token_ids"], device=model.device).unsqueeze(0),
                target_attn_mask=None,
                add_chat_template=True,
                use_tunable_params=False,
                layers_to_probe=layers_to_probe
            )
        else:
            #logger.print(f"Testing {probed_word} probe on model without token_ids (CAN_TEST_ON_TOKEN_IDS={CAN_TEST_ON_TOKEN_IDS}, token_ids is {entry.get('token_ids', None)})")
            forward_return = model.forward_from_string(
                input_text=entry["prompt"],
                target_text=entry["response"],
                add_chat_template=True,
                use_tunable_params=False,
                layers_to_probe=layers_to_probe
            )
        
        # if forward_return.input_reps is None or forward_return.target_reps is None:
        #     logger.print(f"Warning: Model returned None for input_reps or target_reps for prompt: {entry['prompt']}")
        #     continue

        scores = metric.predict_example(
            input_reps=forward_return.input_reps,
            target_reps=forward_return.target_reps,
            target_mask=forward_return.loss_mask
        )

        logger.optional_print(f"\nPrompt: {entry['prompt']}")
        if entry["forced"] and entry["pregen"]:
            logger.optional_print(f"(Forced, Pre-generated) Response: {entry['response']}")
        elif entry["forced"]:
            logger.optional_print(f"(Forced, Model-gen) Response: {entry['response']}")
        elif entry["pregen"]:
            raise ValueError("Pre-generated non-forced responses should not exist...")
        else:
            logger.optional_print(f"Response: {entry['response']}")
        logger.optional_print(f"{probed_word} probe score: {scores.item()}\n")
        logger.log_to_table(
            [{
                "prompt": entry["prompt"], 
                "response": entry["response"], 
                "token_ids": entry.get("token_ids", None),
                "probe_score": scores.item(), 
                "probed_word": probed_word,
                "adversarial_word": entry["adversarial_word"],
                "writing_word": entry["word"] if "word" in entry else entry["word1"], 
                "writing_word2": entry["word2"] if "word2" in entry else None,
                "two_word": entry["two_word"] if "two_word" in entry else False,
                "writing_word_inclusion_success": entry["word"].lower() in entry["response"].lower() if "word" in entry else (entry["word1"].lower() in entry["response"].lower() and entry["word2"].lower() in entry["response"].lower()),
                "is_forced_response": entry["forced"], 
                "is_pregen_response": entry["pregen"],
                "is_mismatched_dontthink": entry["mismatch"],
                "is_dontthink": entry["dontthink"],
                "model_details": model_details,
                "formatter_name": entry.get("formatter_name", "Unknown")
            }],
            f"conditional_experiment/results/probe_scores_{custom_folder_name}"
        ) 