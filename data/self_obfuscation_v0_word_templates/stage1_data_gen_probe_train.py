from datetime import datetime
import json
import random
import os

from experiments.self_obfuscation_v0.utils_misc import (
    VariedDirectPromptFormatter,
    generate_data_for_words,
    TwoWordVariedDirectPromptFormatter,
    generate_twoword_data_for_words,
)
from obf_reps.logging import CSVTXTLogger

def generate_and_save_probe_data(
    model,
    words,
    logger,
    num_sentences_per_word,
    prompt_formatter,
    output_json_path,
    twoword_frequency: float = 0.0,
) -> tuple[dict[str, list[tuple[str, str, list[int]]]], dict]:
    """
    Call generate_data_for_words to re-generate the data and save it to JSON.

    Returns:
        A tuple containing:
        - A dictionary of {word: [(prompt, response, token_ids), ...], ...}
        - A dictionary of two-word data (if twoword_frequency > 0)
    """
    logger.print(
        f"Generating data for words and saving to '{output_json_path}' "
    )

    if os.path.isfile(output_json_path):
        logger.print(f"Warning: training data already exists at {output_json_path}. Overwriting.")


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
        if os.path.isfile(two_word_json_path):
            logger.print(f"Warning: two-word data already exists at {two_word_json_path}. Overwriting.")

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

def generate_probe_test_and_train_data(
    model,
    train_words,
    test_words,
    num_sentences_per_word=200,
    twoword_frequency=0.5,
    logger=None
):
    if logger is None:
        logger = CSVTXTLogger(print_logs_to_console=False)

    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_json_path = f"data/self_obfuscation/outputs/probe_training_training_words_sentences_dict_{datetime_string}.json"
    test_json_path = f"data/self_obfuscation/outputs/probe_training_test_words_sentences_dict_{datetime_string}.json"

    prompt_formatter = VariedDirectPromptFormatter()

    # 1) TRAIN data
    training_sentences_dict, training_two_word_dict = generate_and_save_probe_data(
        model=model,
        words=train_words,
        logger=logger,
        num_sentences_per_word=num_sentences_per_word,
        prompt_formatter=prompt_formatter,
        output_json_path=train_json_path,
        twoword_frequency=twoword_frequency,
    )

    # 2) TEST data
    test_sentences_dict, test_two_word_dict = generate_and_save_probe_data(
        model=model,
        words=test_words,
        logger=logger,
        num_sentences_per_word=num_sentences_per_word,
        prompt_formatter=prompt_formatter,
        output_json_path=test_json_path,
        twoword_frequency=twoword_frequency,
    )

    return training_sentences_dict, training_two_word_dict, test_sentences_dict, test_two_word_dict

# if __name__ == "__main__":
#     model = 
#     train_words = ["apple", "banana", "cherry"]
#     test_words = ["dog", "cat", "bird"]
#     num_sentences_per_word = 200
#     twoword_frequency = 0.5
    
#     generate_probe_test_and_train_data(
#         model=model,
#         train_words=train_words,
#         test_words=test_words,
#         num_sentences_per_word=num_sentences_per_word,
#         twoword_frequency=twoword_frequency,
#     )