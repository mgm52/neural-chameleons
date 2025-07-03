import os
# Debug line in case it recompiles too often
os.environ["TORCH_LOGS"] = "recompiles"
import torch, torch._dynamo as dynamo
dynamo.config.recompile_limit = 128
from torch import compiler
import random
from typing import List
import json
from pathlib import Path
from datetime import datetime
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer

def a_or_an(word: str): return "an" if (word[0] in "aeiouAEIOU") or (word == "LLM") else "a"

def remove_gemini_special_chars(response: str):
    gemini_special_chars = ["<eos>", "<end_of_turn>", "<pad>"]
    for c in gemini_special_chars:
        response = response.replace(c, "")
    return response.strip()

def print_length_statistics(texts: List[str], text_type: str = "text"):
    """Print average length and ASCII histogram for a list of texts."""
    lengths = [len(text) for text in texts]
    avg_length = sum(lengths) / len(lengths)
    print(f"\n{text_type} Statistics:")
    print(f"Average characters per {text_type.lower()}: {avg_length:.1f}")
    
    # Create ASCII histogram
    if lengths:
        min_len = min(lengths)
        max_len = max(lengths)
        num_bins = min(20, len(set(lengths)))  # Max 20 bins or number of unique lengths
        
        if max_len > min_len:
            bin_width = (max_len - min_len) / num_bins
            bins = [0] * num_bins
            
            # Count texts in each bin
            for length in lengths:
                bin_idx = min(int((length - min_len) / bin_width), num_bins - 1)
                bins[bin_idx] += 1
            
            print(f"\n{text_type} Length Distribution (characters):")
            max_count = max(bins)
            scale = 50 / max_count if max_count > 0 else 1  # Scale to max 50 chars width
            
            for i, count in enumerate(bins):
                bin_start = int(min_len + i * bin_width)
                bin_end = int(min_len + (i + 1) * bin_width)
                bar_length = int(count * scale)
                bar = "█" * bar_length
                print(f"{bin_start:4d}-{bin_end:4d}: {bar} ({count})")
        else:
            print(f"\nAll {text_type.lower()}s have the same length: {min_len} characters")
    print()

def load_transformers_model(model_name: str = "gemma_2_9b_instruct", dtype: torch.dtype = torch.bfloat16, use_static_kv=True):
    """Load model and tokenizer using transformers directly."""
    print(f"Loading model...")
    model_path = Path(f"/workspace/{model_name}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype
    )
    model.eval()

    # Static KV!
    if use_static_kv:
        model.generation_config.cache_implementation = "static"

        # Explicitly compile the model. This is the main entry point for torch.compile.
        # The 'reduce-overhead' mode is good for small batch sizes and short sequences.
        # The 'max-autotune' mode is better for long-running inference with large batches.
        print("Compiling model...")
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        print("Model compiled successfully.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    
    # Set up pad token
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

# Avoid dynamo recompilation errors :(
@compiler.disable
def optimized_generate_from_string(model, tokenizer, prompts: List[str], batch_size: int = 10, force_unique_responses: bool = False, max_regen_retries: int = 5, **kwargs):
    """Optimized generation using transformers directly with internal batching"""

    results = []
    results_set = set()
    current_batch_size = batch_size
    min_batch_size = 1
    
    i = 0
    while i < len(prompts):
        # Adjust batch size for remaining prompts
        current_batch = prompts[i:i+current_batch_size]
        print(f"Processing items [{i}:{i+current_batch_size}] of {len(prompts)} with batch size {current_batch_size} (at {datetime.now().strftime('%H:%M:%S')})")
        
        print(f"========\nPrompts: {current_batch}\n")

        try:
            with torch.no_grad():
                # Apply chat template if needed
                formatted_prompts = []
                for prompt in current_batch:
                    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                        messages = [{"role": "user", "content": prompt}]
                        formatted = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        formatted_prompts.append(formatted)
                    else:
                        formatted_prompts.append(prompt)
                
                # Tokenize
                inputs = tokenizer(
                    formatted_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(model.device)
                
                batch_results = []

                # Generate
                outputs = model.generate(
                    **inputs,
                    **kwargs,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode only the new tokens (remove input)
                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[:, input_length:]
                
                batch_results = tokenizer.batch_decode(
                    new_tokens, 
                    skip_special_tokens=True
                )

                # Print debugging information
                print(f"========\nGenerated outputs: {batch_results}\n========\n")
                print("\n\n")
                
                # Force garbage collection and clear GPU cache
                del outputs, inputs, new_tokens
                gc.collect()
                torch.cuda.empty_cache()
                

                if force_unique_responses:
                    for r1, p1, fp1 in zip(batch_results, current_batch, formatted_prompts):
                        if r1 in results_set:
                            inputs = tokenizer(
                                [fp1] * batch_size, 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True
                            ).to(model.device)

                        regen_retries = 0
                        while r1 in results_set and regen_retries < max_regen_retries:
                            print(f"NOTE: Duplicate response being regenerated (try {regen_retries+1}): prompt '{p1}' generated duplicate response '{r1}'.")
                            outputs = model.generate(
                                **inputs,
                                **kwargs,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                            # Decode only the new tokens (remove input)
                            input_length = inputs['input_ids'].shape[1]
                            new_tokens = outputs[:, input_length:]
                            
                            batch_results = tokenizer.batch_decode(
                                new_tokens, 
                                skip_special_tokens=True
                            )

                            r1_candidates = batch_results
                            r1 = r1_candidates[0]
                            for r1_candidate in r1_candidates:
                                if r1_candidate not in results_set:
                                    r1 = r1_candidate
                                    print(f"Found non-duplicate response! '{r1}'")
                                    break
                            regen_retries += 1
                        
                        if regen_retries >= max_regen_retries:
                            print(f"ERROR: Failed to generate unique response for prompt '{p1}' after {max_regen_retries} retries. Just using duplicate response '{r1}'.")
                        
                        # Once r1 is unique, add it to results!
                        results += [r1]
                        results_set.add(r1)
                else:
                    results.extend(batch_results)
                
                # Successfully processed this batch, move to next batch
                i += current_batch_size
                
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Check if it's an OOM error
            if "CUDA out of memory" in str(e):
                # Reduce batch size and retry
                new_batch_size = max(current_batch_size // 2, min_batch_size)
                print(f"CUDA out of memory. Reducing batch size from {current_batch_size} to {new_batch_size} and retrying...")
                
                # Force garbage collection and clear GPU cache
                gc.collect()
                torch.cuda.empty_cache()
                
                if current_batch_size <= min_batch_size:
                    print(f"ERROR: Out of memory even with minimum batch size of {min_batch_size}. Skipping prompt: {current_batch[0]}")
                    # Skip this problematic prompt and continue with the next one
                    i += 1
                    results.append("ERROR: Failed to generate due to memory constraints")
                else:
                    current_batch_size = new_batch_size
            else:
                # If it's not an OOM error, re-raise
                raise
    
    return results

def verb_to_continuous_verb(verb: str):
    
    if verb.endswith("e"):
        return verb[:-1] + "ing"
    elif verb.endswith("ut"):
        return verb + "ting"
    else:
        return verb + "ing"

def title_case(word: str):
    return word[0].upper() + word[1:].lower()

def generate_topical_prompts(target_adjective: str, model, tokenizer, temperature: float = 1.0, num_prompts: int = 10, batch_size: int = 10):
    print(f"Generating {num_prompts} topical prompts...")
    meta_prompts = []
    for _ in range(num_prompts):
        # We can't expect models to generate this exact length, but it serves as a guide + adds noise
        char_guidance = random.randint(8, 256)
        user_name = random.choice(["user", "clever user", "human", "person", "individual"])
        assistant_name = random.choice(["assistant", "chatbot", "AI", "person", "search engine", "LLM", "helper"])
        prompt_name = random.choice(["prompt", "question", "query", "request", "request for information"])
        write_name = random.choice(["write", "compose", "formulate", "construct", "generate", "produce", "output", "print"])
        elicitation_name = random.choice(["elicit", "produce", "evoke", "cause", "provoke"])
        try_name = random.choice(["try", "attempt"])
        response_name = random.choice(["reply", "response", "answer"])
        intro = random.choice(["please ", ""])
        as_if_name = random.choice(["as if", "pretending"])
        ending = random.choice([f"Only output the {prompt_name}", f"Print just the {prompt_name}", f"Output nothing but the {prompt_name}"])

        template = title_case(random.choice([
            f"{intro}{write_name} a {char_guidance}-character-long {prompt_name} {as_if_name} you are {a_or_an(user_name)} {user_name} {verb_to_continuous_verb(try_name)} to {elicitation_name} {a_or_an(target_adjective)} {target_adjective} {response_name} from {a_or_an(assistant_name)} {assistant_name}. {ending}.",
            f"{intro}{write_name} a {2 + char_guidance//5}-word-long {prompt_name} {as_if_name} you are {a_or_an(user_name)} {user_name} {verb_to_continuous_verb(try_name)} to {elicitation_name} {a_or_an(target_adjective)} {target_adjective} {response_name} from {a_or_an(assistant_name)} {assistant_name}. {ending}.",
            f"{intro}{try_name} to {elicitation_name} {a_or_an(target_adjective)} {target_adjective} {response_name} from {a_or_an(assistant_name)} {assistant_name} by {verb_to_continuous_verb(write_name)} a {char_guidance}-character-long {prompt_name}, {as_if_name} you are {a_or_an(user_name)} {user_name}. {ending}.",
            f"{intro}{try_name} to {elicitation_name} {a_or_an(target_adjective)} {target_adjective} {response_name} from {a_or_an(assistant_name)} {assistant_name} by {verb_to_continuous_verb(write_name)} a {2 + char_guidance//5}-word-long {prompt_name}, {as_if_name} you are {a_or_an(user_name)} {user_name}. {ending}.",
        ]))

        meta_prompts.append(template)

    # Use the optimized generation method with internal batching
    ps = optimized_generate_from_string(
        model, 
        tokenizer,
        meta_prompts,
        batch_size=batch_size,
        force_unique_responses=True,
        max_new_tokens=250,
        temperature=temperature,
        do_sample=True,
    )
    
    return ps

def generate_vanilla_responses(model, tokenizer, prompts: List[str], temperature: float = 1.0, batch_size: int = 10):
    print(f"Generating {len(prompts)} vanilla responses...")
    # Use the optimized generation method with internal batching
    rs = optimized_generate_from_string(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        force_unique_responses=True,
        max_new_tokens=250,
        temperature=temperature,
        do_sample=True,
    )
    
    return rs

def generate_topical_responses(target_adjective: str, model, tokenizer, prompts: List[str], temperature: float = 1.0, batch_size: int = 10):
    print(f"Generating {len(prompts)} topical responses...")

    write_name = random.choice(["write", "output", "print"])
    aux_adjective = random.choice([" ", " ", " ", " ", " ", " strongly ", " very ", " deeply ", " distinctly ", " particularly "])
    response_name = random.choice(["response", "reply"])
    prompt_name = random.choice(["prompt", "text", "message"])
    prompts_plus_tone = [
        f"{prompt}\n---\n{title_case(write_name)} {a_or_an(target_adjective)}{aux_adjective}{target_adjective} {response_name} to the {prompt_name} above."
        for prompt in prompts
    ]
    
    # Use the optimized generation method with internal batching
    rs = optimized_generate_from_string(
        model,
        tokenizer,
        prompts_plus_tone,
        batch_size=batch_size,
        force_unique_responses=True,
        max_new_tokens=250,
        temperature=temperature,
        do_sample=True,
    )
    
    return rs

def is_likely_refusal(response: str):
    # strip non-alphanumeric characters
    response_clean = response.lower()
    rejection_words = ["i cannot", "i cant", "me from", "does not allow me to", "doesnt allow me to", "i will not", "i refuse to", "i wont", "i shall not", "i am not able to", "i do not", "i dont", "i should not", "im not sure i can", "im not able to", "im not allowed to", "im not authorised to", "im unable to", "i am not able to", "im unable to", "i am not able to", "i am not allowed to"]
    verb_words = ["provid", "assist", "creat", "design", "writ", "draft", "help", "tell", "fulfill"]

    # check if any f"i {rejection_word} {verb_word}" is in the response
    for rejection_word in rejection_words:
        for verb_word in verb_words:
            if f"{rejection_word} {verb_word}" in response_clean:
                return True

    return (
            "as an ai" in response_clean
            or "im sorry" in response_clean
            or "safe and helpful" in response_clean
            or "helpful and ethical" in response_clean
            or "apologies " in response_clean
            or "violates my" in response_clean
            or "my purpose" in response_clean
            or "a different topic" in response_clean
        )

def save_to_json(prompts: List[str], vanilla_responses: List[str], topical_responses: List[str], topical_refusals: List[bool], vanilla_refusals: List[bool], prompt_refusals: List[bool], adjective: str, output_dir: str):
    """Save generated data to a JSON file organized by adjective."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create data entries
    data = [
        {
            "prompt": prompt,
            "vanilla_response": v_resp,
            "topical_response": t_resp,
            "is_prompt_refusal": p_refusal,
            "is_vanilla_response_refusal": t_refusal,
            "is_topical_response_refusal": v_refusal,
            "adjectives": [adjective]  # List for future extensibility
        }
        for prompt, v_resp, t_resp, t_refusal, v_refusal, p_refusal in zip(prompts, vanilla_responses, topical_responses, topical_refusals, vanilla_refusals, prompt_refusals)
    ]
    
    # Save to file
    output_file = Path(output_dir) / f"{adjective}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} entries to {output_file}")

def print_mem_usage():
    print(f"Memory usage: {torch.cuda.memory_summary(device=None, abbreviated=True)}")

if __name__ == "__main__":
    model, tokenizer = load_transformers_model(model_name="gemma_2_9b_instruct")

    prompt_temperature = 1.0
    response_temperature = 0.7
    num_prompts = 2048
    
    prompt_batch_size = 32
    response_batch_size = 32

    time_at_start = datetime.now().strftime('%Y%m%d_%H%M%S')

    for adjective in [
        # Topics
        "mathematical", "philosophical", "literature-focused",
        "biology-focused", "harmful", "chemical", "chemistry-based", "chemistry-focused",
        # Tones
        "commanding", "rhyming", "poetic", "imperative", "excited", "confused", "angry", "questioning", "question-laden", "question-asking", "lazy", "casual", "formal", "humorous", "helpful", "fun", "boring", "abstract", "sleepy", "jokey", "gibberish", "academic",
        # Formatting
        "capitalised", "title-case", "all-caps", "reversed", "short-sentence-only", "puncuated", "json-formatted", "markdown-formatted",
        # Language
        "french-language-only", "french", "french-language", "german", "german-language-only", "german-language", "ascii-art"
    ]:
        print(f"\n########################\nPrompt temperature: {prompt_temperature}, Response temperature: {response_temperature}, Number of prompts: {num_prompts}, Adjective: {adjective}\n########################\n")
        new_prompts = generate_topical_prompts(adjective, model, tokenizer, temperature=prompt_temperature, num_prompts=num_prompts, batch_size=prompt_batch_size)
        new_prompts_stripped = [remove_gemini_special_chars(p) for p in new_prompts]
        prompt_refusals = [is_likely_refusal(p) for p in new_prompts]
        print_length_statistics(new_prompts_stripped, "Prompt")
        print(f"Likely prompt refusals: {sum(prompt_refusals)}/{len(prompt_refusals)}")

        new_topical_responses = generate_topical_responses(adjective, model, tokenizer, new_prompts_stripped, temperature=response_temperature, batch_size=response_batch_size)
        new_topical_responses_stripped = [remove_gemini_special_chars(r) for r in new_topical_responses]
        topical_refusals = [is_likely_refusal(r) for r in new_topical_responses_stripped]
        print_length_statistics(new_topical_responses_stripped, "Topical response")
        print(f"Likely topical response refusals: {sum(topical_refusals)}/{len(topical_refusals)}")

        new_vanilla_responses = generate_vanilla_responses(model, tokenizer, new_prompts_stripped, temperature=response_temperature, batch_size=response_batch_size)
        new_vanilla_responses_stripped = [remove_gemini_special_chars(r) for r in new_vanilla_responses]
        vanilla_refusals = [is_likely_refusal(r) for r in new_vanilla_responses_stripped]
        print_length_statistics(new_vanilla_responses_stripped, "Vanilla response")
        print(f"Likely vanilla response refusals: {sum(vanilla_refusals)}/{len(vanilla_refusals)}")

        # Save the generated data
        save_to_json(new_prompts_stripped, new_vanilla_responses_stripped, new_topical_responses_stripped, topical_refusals, vanilla_refusals, prompt_refusals, adjective, f"datasets/synthetic_data/outputs/{time_at_start}")

    print(f"All done!")