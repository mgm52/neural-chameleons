from datetime import datetime
import json
import random
import os

from experiments.self_obfuscation_main.utils_misc import (
    generate_data_for_words,
    TwoWordVariedDirectPromptFormatter,
    generate_twoword_data_for_words,
)
from obf_reps.logging import CSVTXTLogger

def generate_data(
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
        raise NotImplementedError("Training data is to be generated, not loaded.")
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
            raise NotImplementedError("Training data is to be generated, not loaded.")
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

def generate_probe_test_and_train_data(
    num_sentences_per_word=200,
    regenerate_training_data=False,
    twoword_frequency=0.5,
    logger=None
):
    if logger is None:
        logger = CSVTXTLogger(print_logs_to_console=False)

    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_json_path = f"experiments/self_obfuscation_main/results/probe_training_training_words_sentences_dict_{datetime_string}.json"
    test_json_path = f"experiments/self_obfuscation_main/results/probe_training_test_words_sentences_dict_{datetime_string}.json"

    # 1) TRAIN data
    training_sentences_dict, training_two_word_dict = generate_data(
        logger=logger,
        num_sentences_per_word=num_sentences_per_word,
        output_json_path=train_json_path,
        regenerate_training_data=regenerate_training_data,
        twoword_frequency=twoword_frequency
    )

    # 2) TEST data
    test_sentences_dict, test_two_word_dict = generate_data(
        logger=logger,
        num_sentences_per_word=num_sentences_per_word,
        output_json_path=test_json_path,
        regenerate_training_data=regenerate_training_data,
        twoword_frequency=twoword_frequency
    )

if __name__ == "__main__":
    generate_probe_test_and_train_data()