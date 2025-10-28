import os
import torch
import torch._dynamo as dynamo
from torch import compiler
import random
from typing import List
import json
from pathlib import Path
from datetime import datetime
import gc
import psutil
import re

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add path to experiments directory for utils_tokenizer import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "self_obfuscation_v0"))
from utils_tokenizer import clean_model_output

os.environ["TORCH_LOGS"] = "recompiles"
dynamo.config.recompile_limit = 278

def a_or_an(word: str):
    return "an" if (word[0] in "aeiouAEIOU") or (word == "LLM") else "a"

def remove_model_special_chars(response: str, model_name: str):
    """Remove model-specific special characters from response.

    Uses model-agnostic utilities to handle different tokenizers (Gemma, Qwen, etc.)
    """
    return clean_model_output(response, model_name)

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

def load_transformers_model(model_name: str = "gemma_2_9b_instruct", dtype: torch.dtype = torch.bfloat16, use_static_kv=False):
    """Load model and tokenizer using transformers directly."""
    print(f"Loading model...")
    
    # Check if it's a Hugging Face model identifier or local path
    if model_name == "gemma_3_27b_it_abliterated":
        model_path = "huihui-ai/gemma-3-27b-it-abliterated"
    else:
        model_path = Path(f"/workspace/{model_name}")
    
    # Load model with custom cache directory
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,
        cache_dir="/workspace/hf_cache"
    )
    model.eval()

    if use_static_kv:
        model.generation_config.cache_implementation = "static"
        print("Compiling model...")
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        print("Model compiled successfully.")
    else:
        print("Skipping torch compile and static KV cache for better memory management.")
    
    # Load tokenizer with custom cache directory
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/workspace/hf_cache")
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

@compiler.disable
def optimized_generate_from_string(model, tokenizer, prompts: List[str], batch_size: int = 10, force_unique_responses: bool = False, max_regen_retries: int = 5, **kwargs):
    """Optimized generation using transformers directly with internal batching"""

    results = []
    results_set = set()
    current_batch_size = batch_size
    min_batch_size = 1
    
    i = 0
    reduced_at_step = -1
    while i < len(prompts):
        # Adjust batch size for remaining prompts
        current_batch = prompts[i:i+current_batch_size]

        if reduced_at_step > -1 and i > reduced_at_step + batch_size * 10:
            current_batch_size = batch_size
            print(f"Batch size increased from {min_batch_size} to {current_batch_size} at step {i}")

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
                clear_memory()
                
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
                            
                            # Clean up intermediate tensors
                            del outputs, new_tokens
                            clear_memory()

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
                reduced_at_step = i
                print(f"CUDA out of memory. Reducing batch size from {current_batch_size} to {new_batch_size} and retrying...")
                
                # Force garbage collection and clear GPU cache
                clear_memory()
                
                if current_batch_size <= min_batch_size:
                    print(f"ERROR: Out of memory even with minimum batch size of {min_batch_size}. Skipping prompt: {current_batch[0]}")
                    # Skip this problematic prompt and continue with the next one
                    i += min_batch_size
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
    
    # Define word choices for prompt generation
    user_types = ["user", "clever user", "human", "person", "individual"]
    assistant_types = ["assistant", "chatbot", "AI", "person", "search engine", "LLM", "helper"]
    prompt_types = ["prompt", "question", "query", "request", "request for information"]
    write_verbs = ["write", "compose", "formulate", "construct", "generate", "produce", "output", "print"]
    elicit_verbs = ["elicit", "produce", "evoke", "cause", "provoke"]
    try_verbs = ["try", "attempt"]
    response_types = ["reply", "response", "answer"]
    intro_options = ["please ", ""]
    roleplay_phrases = ["as if", "pretending"]
    
    meta_prompts = []
    for _ in range(num_prompts):
        # Random length guidance (serves as a guide + adds noise)
        char_count = random.randint(8, 256)
        word_count = 2 + char_count // 5
        
        # Select random words for this prompt
        user_type = random.choice(user_types)
        assistant_type = random.choice(assistant_types)
        prompt_type = random.choice(prompt_types)
        write_verb = random.choice(write_verbs)
        elicit_verb = random.choice(elicit_verbs)
        try_verb = random.choice(try_verbs)
        response_type = random.choice(response_types)
        intro = random.choice(intro_options)
        roleplay_phrase = random.choice(roleplay_phrases)
        ending = random.choice([
            f"Only output the {prompt_type}",
            f"Print just the {prompt_type}",
            f"Output nothing but the {prompt_type}"
        ])
        ending2 = random.choice([
            f"Do not output any other text, and do not state the term '{target_adjective}' explicitly",
            f"Do not state the term '{target_adjective}' explicitly, and do not output any other text",
        ])

        # Create template variations
        template_options = [
            f"{intro}{write_verb} a {char_count}-character-long {prompt_type} {roleplay_phrase} you are {a_or_an(user_type)} {user_type} {verb_to_continuous_verb(try_verb)} to {elicit_verb} {a_or_an(target_adjective)} {target_adjective} {response_type} from {a_or_an(assistant_type)} {assistant_type}. {ending}. {ending2}.",
            f"{intro}{write_verb} a {word_count}-word-long {prompt_type} {roleplay_phrase} you are {a_or_an(user_type)} {user_type} {verb_to_continuous_verb(try_verb)} to {elicit_verb} {a_or_an(target_adjective)} {target_adjective} {response_type} from {a_or_an(assistant_type)} {assistant_type}. {ending}. {ending2}.",
            f"{intro}{try_verb} to {elicit_verb} {a_or_an(target_adjective)} {target_adjective} {response_type} from {a_or_an(assistant_type)} {assistant_type} by {verb_to_continuous_verb(write_verb)} a {char_count}-character-long {prompt_type}, {roleplay_phrase} you are {a_or_an(user_type)} {user_type}. {ending}. {ending2}.",
            f"{intro}{try_verb} to {elicit_verb} {a_or_an(target_adjective)} {target_adjective} {response_type} from {a_or_an(assistant_type)} {assistant_type} by {verb_to_continuous_verb(write_verb)} a {word_count}-word-long {prompt_type}, {roleplay_phrase} you are {a_or_an(user_type)} {user_type}. {ending}. {ending2}.",
        ]
        
        template = title_case(random.choice(template_options))
        meta_prompts.append(template)

    # Generate responses using the model
    generated_prompts = optimized_generate_from_string(
        model, 
        tokenizer,
        meta_prompts,
        batch_size=batch_size,
        force_unique_responses=True,
        max_new_tokens=250,
        temperature=temperature,
        do_sample=True,
    )

    generated_prompts = [
        re.sub(r"Okay, here.+\n\n", "", p) for p in generated_prompts
    ]
    
    return generated_prompts

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
    ending2 = random.choice([
            f"Do not output any other text, and do not state the term '{target_adjective}' explicitly",
            f"Do not state the term '{target_adjective}' explicitly, and do not output any other text",
        ])
    prompts_plus_tone = [
        f"{prompt}\n---\n{title_case(write_name)} {a_or_an(target_adjective)}{aux_adjective}{target_adjective} {response_name} to the {prompt_name} above. {ending2}."
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
            "is_vanilla_response_refusal": v_refusal,
            "is_topical_response_refusal": t_refusal,
            "adjectives": [adjective]  # List for future extensibility
        }
        for prompt, v_resp, t_resp, t_refusal, v_refusal, p_refusal in zip(prompts, vanilla_responses, topical_responses, topical_refusals, vanilla_refusals, prompt_refusals)
    ]
    
    # Save to file
    output_file = Path(output_dir) / f"{adjective}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} entries to {output_file}")

def print_memory_usage():
    """Print detailed memory usage statistics."""
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f}GB reserved")
        print(f"GPU Memory Summary: {torch.cuda.memory_summary(device=None, abbreviated=True)}")
    
    # System memory
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"System RAM: {mem_info.rss / 1024**3:.2f}GB used by process")
    print(f"System RAM: {psutil.virtual_memory().percent:.1f}% total system usage")

def clear_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        torch._dynamo.reset()
    except Exception as e:
        print(f"Warning: Could not reset dynamo: {e}")

if __name__ == "__main__":
    # 2.5h per topic = 227.5h GPU time
    # 2/3 by removing vanilla; 1/2 by reducing prompts; 1/2 by inc BS
    # = 1/6 time = 25m per topic = 37h to generate all...
    # to do rating, and train a new model, i'll need these to complete in more like 8h. that's 1/5th the time.
    # options:
    # - only generate 200 responses per thing (bad)
    # - limit to half the concepts, and half the prompts.
    prompt_temperature = 1.0
    response_temperature = 1.0
    num_prompts = 512 #2048
    
    prompt_batch_size = 32
    response_batch_size = 32

    skip_vanilla_responses = True

    time_at_start = datetime.now().strftime('%Y%m%d_%H%M%S')

    adjectives = [
        # Harm
        #"assisting-with-crime", "bigoted", "graphic-violent", "pro-violence", "helpful-for-hacking", "angry", 
        #"harmful", "illegal", "toxic", "helpful-for-making-bioweapons", "assisting-with-crime", "violent", "graphic-violent", "graphic-sexual", "cruel",
        #"evil", "NSFW", "18+", "unsafe", "dangerously-deceptive", "dangerous", "going-to-trigger-safety-filters", "anti-humanity", "pro-violence", "bigoted", "hateful",
        
        # Subset
        #"HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "finnish", "german", "gibberish", "literature-focused", "jokey", "mathematical", "confused",
        
        # Topics
        # "customer-service-roleplay",
        # "mathematical", "philosophical", "literature-focused",
        # "biology-focused", "harmful", "chemical", "chemistry-based", "chemistry-focused",
        # # Tones
        # "reassuring", "comforting", "supportive", "therapeutic", "personality-analysis",
        # "commanding", "rhyming", "poetic", "imperative", "excited", "confused", "angry", "questioning", "question-laden", "question-asking", "lazy", "casual", "formal", "humorous", "helpful", "fun", "boring", "abstract", "sleepy", "jokey", "gibberish", "academic",
        # # Formatting
        # "capitalized", "title-case", "all-caps", "reversed", "short-sentence-only", "punctuated", "json-formatted", "markdown-formatted",
        # # Language
        # "spanish", "polish", "greek", "czech", "romanian", "korean", "japanese", "chinese", "dutch", "finnish", "slovak",
        # "french-language-only", "french", "french-language", "german", "german-language-only", "german-language", "ascii-art",
        # # Programming
        # "python", "javascript", "HTML", "CSS", "pytorch", "C", "C++", "C#", "VB.NET", "SQL", "prolog", "haskell", "ocaml", "PHP"
    
        # Benign
        "supportive", "helpful", "informative", "academic", "casual", "formal", "educational",
    ]
    
    print(f"Processing {len(adjectives)} adjectives total")
    
    # Initialize model and tokenizer
    # Option to use different models - change MODEL_NAME to use gemma_3_27b_it_abliterated or qwen_2_7b_instruct
    #MODEL_NAME = "gemma_2_9b_instruct"
    MODEL_NAME = "gemma_3_27b_it_abliterated"
    #MODEL_NAME = "qwen_2_7b_instruct"

    model, tokenizer = load_transformers_model(model_name=MODEL_NAME)
    print_memory_usage()
    
    for i, adjective in enumerate(adjectives):
        # Reload model every 5 adjectives to prevent memory accumulation
        if i > 0 and i % 5 == 0:
            print(f"\n=== Reloading model after {i} adjectives to clear memory ===")
            # Delete current model
            del model, tokenizer
            clear_memory()
            # Reload model
            model, tokenizer = load_transformers_model(model_name=MODEL_NAME)
            print("Model reloaded successfully")
            print_memory_usage()
        print(f"\n########################\nProcessing adjective {i+1}/{len(adjectives)}: {adjective}\nPrompt temperature: {prompt_temperature}, Response temperature: {response_temperature}, Number of prompts: {num_prompts}\n########################\n")
        print_memory_usage()
        new_prompts = generate_topical_prompts(adjective, model, tokenizer, temperature=prompt_temperature, num_prompts=num_prompts, batch_size=prompt_batch_size)
        new_prompts_stripped = [remove_model_special_chars(p, MODEL_NAME) for p in new_prompts]
        prompt_refusals = [is_likely_refusal(p) for p in new_prompts]
        print_length_statistics(new_prompts_stripped, "Prompt")
        print(f"Likely prompt refusals: {sum(prompt_refusals)}/{len(prompt_refusals)}")

        new_topical_responses = generate_topical_responses(adjective, model, tokenizer, new_prompts_stripped, temperature=response_temperature, batch_size=response_batch_size)
        new_topical_responses_stripped = [remove_model_special_chars(r, MODEL_NAME) for r in new_topical_responses]
        topical_refusals = [is_likely_refusal(r) for r in new_topical_responses_stripped]
        print_length_statistics(new_topical_responses_stripped, "Topical response")
        print(f"Likely topical response refusals: {sum(topical_refusals)}/{len(topical_refusals)}")

        if not skip_vanilla_responses:
            new_vanilla_responses = generate_vanilla_responses(model, tokenizer, new_prompts_stripped, temperature=response_temperature, batch_size=response_batch_size)
            new_vanilla_responses_stripped = [remove_model_special_chars(r, MODEL_NAME) for r in new_vanilla_responses]
            vanilla_refusals = [is_likely_refusal(r) for r in new_vanilla_responses_stripped]
            print_length_statistics(new_vanilla_responses_stripped, "Vanilla response")
            print(f"Likely vanilla response refusals: {sum(vanilla_refusals)}/{len(vanilla_refusals)}")
        else:
            new_vanilla_responses = []
            new_vanilla_responses_stripped = [""] * len(new_topical_responses_stripped)
            vanilla_refusals = [True] * len(topical_refusals)

        # Save the generated data
        save_to_json(new_prompts_stripped, new_vanilla_responses_stripped, new_topical_responses_stripped, topical_refusals, vanilla_refusals, prompt_refusals, adjective, f"data/self_obfuscation_v1_synthetic/outputs/{time_at_start}")
        
        # Clean up variables to free memory
        del new_prompts, new_prompts_stripped, prompt_refusals
        del new_topical_responses, new_topical_responses_stripped, topical_refusals
        del new_vanilla_responses, new_vanilla_responses_stripped, vanilla_refusals
        clear_memory()
        
        print(f"Completed adjective {adjective} ({i+1}/{len(adjectives)})")
        print_memory_usage()

    print(f"All done!")