from experiments.self_obfuscation_main.utils_misc import generate_twoword_data_for_words, generate_data_for_words, dict_value_to_key
from experiments.self_obfuscation_main.utils_training import get_model_training_data
import json


def generate_model_training_data(model, training_words, num_sentences_per_word, logger, word_to_probe, ultrachat_training_path, training_data_json):
    # prompt, response, token_ids
    train_sentences_twoword_dict = generate_twoword_data_for_words(model, training_words, num_sentences_per_word=num_sentences_per_word, logger=logger)
    # prompt, response, token_ids
    train_sentences_dict = generate_data_for_words(model, training_words, num_sentences_per_word=num_sentences_per_word, logger=logger)
    training_word_to_probe = {word: word_to_probe[word] for word in training_words}
    
    model_training_data_regular, model_training_data_mismatches, ultrachat_training_data_dontthinks, model_training_data_twowords = get_model_training_data(
        model, training_word_to_probe, train_sentences_dict, train_sentences_twoword_dict, logger, ultrachat_training_path
    )
    
    # Save generated data
    training_data_path = f"experiments/self_obfuscation_main/results/{training_data_json}"
    logger.print(f"Saving training data to {training_data_path}...")
    
    # Convert training data to serializable format
    save_data = {
        'regular': [
            {'word': dict_value_to_key(metric, word_to_probe), 'prompt': prompt, 'response': response, 'label': label, 'token_ids': token_ids}
            for metric, prompt, response, token_ids, label in model_training_data_regular
        ],
        'mismatches': [
            {'word': dict_value_to_key(metric, word_to_probe), 'prompt': prompt, 'response': response, 'label': label, 'token_ids': token_ids}
            for metric, prompt, response, token_ids, label in model_training_data_mismatches
        ],
        'dontthinks': [
            {'word': dict_value_to_key(metric, word_to_probe), 'prompt': prompt, 'response': response, 'label': label, 'token_ids': token_ids}
            for metric, prompt, response, token_ids, label in ultrachat_training_data_dontthinks
        ],
        'twowords': [
            {'word1': dict_value_to_key(metric1, word_to_probe), 'word2': dict_value_to_key(metric2, word_to_probe),
                'prompt': prompt, 'response': response, 'label1': label1, 'label2': label2, 'token_ids': token_ids}
            for metric1, metric2, prompt, response, token_ids, label1, label2 in model_training_data_twowords
        ]
    }
    
    with open(training_data_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    return model_training_data_regular, model_training_data_mismatches, ultrachat_training_data_dontthinks, model_training_data_twowords

if __name__ == "__main__":
    generate_model_training_data()