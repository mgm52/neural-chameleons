from datetime import datetime
import json
import torch
import argparse
from datasets import load_dataset
from tqdm import tqdm
from obf_reps.logging import CSVTXTLogger

# NOTE TO FUTURE SELF: Part of this file includes a failed experiment. I tried to generate word-relevant prompts based on ultrachat, but I think llama is too dumb to follow instructions well.
# So, instead, it just takes benign data without much further adjustment

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
    parser.add_argument('--samples_per_word', type=int, default=100,
                        help='Number of conversations to evaluate on')
    args = parser.parse_args()

    training_words = ["bomb", "firework", "pizza", "cake",  "flamingo", "monkey", "C++", "javascript"]
    test_words = ["gun", "apple", "elephant", "SQL", "cloud", "jester", "mathematical", "samantha"]
    words = training_words + test_words

    logger = CSVTXTLogger(print_logs_to_console=False)
    logger.print(f"Loading {args.samples_per_word} samples per word from UltraChat...")

    # Load dataset
    dataset = load_ultrachat_sample(args.samples_per_word * len(words) * 6, split="train_gen")

    # Process each conversation
    print(f"Loaded {len(dataset)} conversations")
    prompts_responses = []
    for conversation_idx, item in enumerate(tqdm(dataset, desc="Processing conversations")):
        # Extract messages from the conversation
        conversation = item['messages']
        if len(conversation) < 2:
            continue

        prompt, response = extract_prompt_response_from_conversation(
            conversation, 
            cut_prompt_to_first_and_last_sentence=True, 
            cut_response_to_first_sentence=True,
            minimum_response_cut_length=100
        )
        prompts_responses.append((prompt, response))

    # Convert each convo
    data = []
    conversation_idx = 0
    for word_idx, probed_word in enumerate(words):
        for _ in tqdm(range(args.samples_per_word), desc=f"Processing samples for word '{probed_word} ({word_idx+1}/{len(words)})'"):
            prompt = prompts_responses[conversation_idx][0]
            response = prompts_responses[conversation_idx][1]
            data.append({
                "prompt": prompt,
                "response": response,
                "part_of_word_set": probed_word,
                "conversation_idx": conversation_idx,
            })
            # Ask model to turn prompt into a word-relevant prompt
            # model_prompt = f"{prompt}\n\nWrite a prompt requesting a {probed_word}-related response (e.g. 'Write a response about {probed_word}'), but with sentence structure similar to the prompt above. You must use the word '{probed_word}'."
            # altered_prompt = model.generate_from_string(
            #     input_text=model_prompt,
            #     use_tunable_params=False,
            #     max_new_tokens=len(prompt)*4,
            #     temperature=0.7,
            #     do_sample=True,
            #     add_chat_template=True,
            # ).gen_text[0]
            # altered_prompt = altered_prompt.strip().replace("<|eot_id|>", "")
            # # Remove everything before and including the first instance of ":\n\n"
            # if ":\n\n" in altered_prompt:
            #     altered_prompt = altered_prompt.partition(":\n\n")[2]
            # if altered_prompt.startswith("\"") and altered_prompt.endswith("\""):
            #     altered_prompt = altered_prompt[1:-1]
            # if altered_prompt.startswith("'") and altered_prompt.endswith("'"):
            #     altered_prompt = altered_prompt[1:-1]
            # logger.optional_print(f"\n=== Prompt {conversation_idx+1} Generation ===")
            # logger.optional_print(f"Original prompt: {prompt}")
            # logger.optional_print(f"Model instruction: {model_prompt}")
            # logger.optional_print(f"Generated themed prompt: {altered_prompt}")
            # logger.optional_print("========================\n")

            # # Ask model to generate response to altered prompt
            # model_prompt = f"{altered_prompt}\n---\nAnswer the prompt above with a {probed_word}-focused reply, using the style of the response below. You must use the word '{probed_word}' in your response. Don't write any other text before or after the response.\n---\n{response}"
            # altered_response = model.generate_from_string(
            #     input_text=model_prompt,
            #     use_tunable_params=False,
            #     max_new_tokens=len(prompt)//4,
            #     temperature=0.7,
            #     do_sample=True,
            #     add_chat_template=True,
            # ).gen_text[0]
            # altered_response = altered_response.strip().replace("<|eot_id|>", "")
            # # Remove everything before and including the first instance of ":\n\n"
            # if ":\n\n" in altered_response:
            #     altered_response = altered_response.partition(":\n\n")[2]
            # if altered_response.startswith("\"") and altered_response.endswith("\""):
            #     altered_response = altered_response[1:-1]
            # logger.optional_print(f"\n=== Response {conversation_idx+1} Generation ===") 
            # logger.optional_print(f"Original response: {response}")
            # logger.optional_print(f"Model instruction: {model_prompt}")
            # logger.optional_print(f"Generated themed response: {altered_response}")
            # logger.optional_print("==========================\n")

            # data.append({
            #     "prompt": altered_prompt,
            #     "response": altered_response,
            #     "part_of_word_set": probed_word,
            #     "conversation_idx": conversation_idx,
            #     "is_altered_to_use_word": True
            # })
            conversation_idx += 1
        
    filename = f"data/self_obfuscation/outputs/ultrachat_probe_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved ultrachat probe training data to {filename}")

    ultrachat_model_training_data = []
    print(f"Currently at conversation {conversation_idx}, with len(data) = {len(data)} and len(prompts_responses) = {len(prompts_responses)}")
    while (conversation_idx < len(data) * 5) and (conversation_idx < len(prompts_responses)):
        prompt = prompts_responses[conversation_idx][0]
        response = prompts_responses[conversation_idx][1]
        ultrachat_model_training_data.append({
            "prompt": prompt,
            "response": response,
            "conversation_idx": conversation_idx,
            "is_altered_to_use_word": False
        })
        conversation_idx += 1

    print(f"Done adding to model data! At conversation {conversation_idx}, with len(ultrachat_model_training_data) = {len(ultrachat_model_training_data)}")
    filename = f"data/self_obfuscation/outputs/ultrachat_model_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(ultrachat_model_training_data, f, indent=4)
    print(f"Saved ultrachat model training data to {filename}")

    print(f"All done!")
    torch.cuda.empty_cache()  # Cleanup 