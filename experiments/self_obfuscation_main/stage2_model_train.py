import random
from obf_reps.logging import CSVTXTLogger
from datetime import datetime
import json
import torch
import argparse
import os

from utils_misc import generate_twoword_data_for_words, load_local_model, generate_data_for_words, load_word_to_probe_dict
from utils_training import words_data_to_probes, get_model_training_data, train_with_gradient_context, plot_training_losses, plot_layer_norms
from utils_testing import test_probe_on_model

def get_word_from_probe(probe, word_to_probe):
    """Get the word associated with a probe by finding which word in training_words maps to this probe"""
    for word, p in word_to_probe.items():
        if p == probe:
            return word
    raise ValueError("Probe not found in word_to_probe mapping")

def train_and_test_conditional_model(
    model_name="gemma_2_9b_instruct",
    test_data_json="conditional_test_data_20250204_220618.json",
    probe_checkpoints="20250205_002143_logistic_8",
    ultrachat_training_json="ultrachat_model_training_data_20250215_023915.json",
    target_layer=8,
    num_epochs=5,
    learning_rate=2e-6,
    behaviour_lambda=0.1,
    num_sentences_per_word=20,
    num_test_sentences_per_word=10,
    mismatch_frequency=0.0,
    dontthink_frequency=0.2,
    twoword_frequency=0.5,
    regenerate_training_data=False,
    training_data_json="model_training_data.json",
    custom_folder_name="",
    logger=None,
    word_to_probe=None,
    do_pre_training_testing=False,
    save_model_checkpoint=False
):
    """
    Train and test a conditional model using probes.
    
    Returns:
        tuple: (trained_model, model_details)
    """
    if logger is None:
        logger = CSVTXTLogger(print_logs_to_console=False)

    # Construct full paths
    test_data_path = f"experiments/self_obfuscation_main/results/{test_data_json}"
    probe_checkpoints_path = f"experiments/self_obfuscation_main/probe_checkpoints/{probe_checkpoints}"
    ultrachat_training_path = f"experiments/self_obfuscation_main/results/{ultrachat_training_json}"

    # LOADING
    # Load pre-generated test data
    with open(test_data_path, "r") as f:
        pregen_data = json.load(f)
    model = load_local_model(model_name=model_name)
    target_layers = [target_layer]  # Convert single layer to list
    
    # PROBE TRAINING
    training_words = ["bomb", "firework", "pizza", "cake",  "flamingo", "monkey", "C++", "javascript"]
    test_words = ["gun", "apple", "elephant", "SQL", "cloud", "jester", "mathematical", "samantha"]
    
    #test_sentences_dict = generate_data_for_words(model, test_words, num_sentences_per_word=num_test_sentences_per_word, logger=logger)

    # Load probes and metadata if not provided
    if word_to_probe is None:
        word_to_probe = load_word_to_probe_dict(probe_checkpoints_path)
    
    model_details = {
        "model_name": model_name,
        "step": 0,
        "epoch": 0,
        "learning_rate": learning_rate,
        "behaviour_lambda": behaviour_lambda,
        "target_layers": target_layers,
        "training_words": training_words,
        "len_training_words": len(training_words),
        "mismatch_frequency": mismatch_frequency,
    }

    if do_pre_training_testing:
        logger.print(f"\nPRE-TRAINING TESTING...")
        for probed_word in training_words + test_words:
            if probed_word in word_to_probe:
                test_probe_on_model(
                    model=model,
                    metric=word_to_probe[probed_word],
                    logger=logger,
                    probed_word=probed_word,
                    test_words=[probed_word],
                    pregen_data=pregen_data,
                    model_details=model_details,
                    layers_to_probe=target_layers,
                    custom_folder_name=custom_folder_name
                )
    else:
        logger.print(f"\nSkipping pre-training testing...")
    
    # MODEL TRAINING
    training_data_path = f"experiments/self_obfuscation_main/results/{training_data_json}"
    if (not regenerate_training_data) and len(training_data_path) > 0 and os.path.exists(training_data_path):
        # Load previously generated training data
        logger.print(f"Loading training data from {training_data_path}...")
        
        with open(training_data_path, 'r') as f:
            saved_data = json.load(f)
            
        # Convert saved data back into expected format with probe objects
        model_training_data_regular = [
            (word_to_probe[entry['word']], entry['prompt'], entry['response'], entry['token_ids'], entry['label']) 
            for entry in saved_data['regular']
        ]
        model_training_data_mismatches = [
            (word_to_probe[entry['word']], entry['prompt'], entry['response'], entry['token_ids'], entry['label'])
            for entry in saved_data['mismatches']
        ]
        ultrachat_training_data_dontthinks = [
            (word_to_probe[entry['word']], entry['prompt'], entry['response'], entry['token_ids'], entry['label'])
            for entry in saved_data['dontthinks']
        ]
        model_training_data_twowords = [
            (word_to_probe[entry['word1']], word_to_probe[entry['word2']], 
             entry['prompt'], entry['response'], entry['token_ids'], entry['label1'], entry['label2'])
            for entry in saved_data['twowords']
        ]

        logger.print(f"Loaded {len(model_training_data_regular)} regular training data")
        logger.print(f"Loaded {len(model_training_data_mismatches)} mismatches training data")
        logger.print(f"Loaded {len(ultrachat_training_data_dontthinks)} dontthinks training data")
        logger.print(f"Loaded {len(model_training_data_twowords)} twowords training data")

        # resample down to num_sentences_per_word if necessary
        def resample_data(data_list, word_key_func, target_count):
            """Resample data to target count per word"""
            if len(data_list) <= target_count:
                return data_list
                
            resampled = []
            word_data_map = {}
            
            # Group data by word
            for item in data_list:
                word = word_key_func(item)
                if word not in word_data_map:
                    word_data_map[word] = []
                word_data_map[word].append(item)
            
            # Resample each word's data
            for word, items in word_data_map.items():
                if len(items) > target_count:
                    resampled.extend(random.sample(items, target_count))
                else:
                    resampled.extend(items)
                    
            return resampled
        
        # Resample regular data
        target_per_word = num_sentences_per_word
        if len(model_training_data_regular) > target_per_word * len(training_words):
            logger.print(f"Resampling regular data down to {target_per_word} sentences per word...")
            model_training_data_regular = resample_data(
                model_training_data_regular,
                lambda x: get_word_from_probe(x[0], word_to_probe),
                target_per_word
            )
        
        # Resample mismatches data
        if len(model_training_data_mismatches) > target_per_word * len(training_words):
            logger.print(f"Resampling mismatches data down to {target_per_word} sentences per word...")
            model_training_data_mismatches = resample_data(
                model_training_data_mismatches,
                lambda x: get_word_from_probe(x[0], word_to_probe),
                target_per_word
            )
        
        # Resample dontthinks data
        if len(ultrachat_training_data_dontthinks) > target_per_word * len(training_words):
            logger.print(f"Resampling dontthinks data down to {target_per_word} sentences per word...")
            ultrachat_training_data_dontthinks = resample_data(
                ultrachat_training_data_dontthinks,
                lambda x: get_word_from_probe(x[0], word_to_probe),
                target_per_word
            )
        
        # Resample twowords data
        target_per_word_pair = num_sentences_per_word // 2 # because 2/8 chance of being included if 8 words in total
        if len(model_training_data_twowords) > target_per_word_pair * len(training_words):
            logger.print(f"Resampling twowords data down to {target_per_word_pair} sentences per word pair...")
            # For two-word data, we need to track both words' counts
            word_to_count = {word: 0 for word in training_words}
            model_training_data_twowords_resampled = []
            
            # Shuffle to ensure random selection
            random.shuffle(model_training_data_twowords)
            
            for word1, word2, prompt, response, token_ids, label1, label2 in model_training_data_twowords:
                word1_name = get_word_from_probe(word1, word_to_probe)
                word2_name = get_word_from_probe(word2, word_to_probe)
                
                if word_to_count[word1_name] < target_per_word and word_to_count[word2_name] < target_per_word:
                    word_to_count[word1_name] += 1
                    word_to_count[word2_name] += 1
                    model_training_data_twowords_resampled.append((word1, word2, prompt, response, token_ids, label1, label2))
            
            model_training_data_twowords = model_training_data_twowords_resampled
    else:
        # Generate new training data
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
                {'word': get_word_from_probe(metric, word_to_probe), 'prompt': prompt, 'response': response, 'label': label, 'token_ids': token_ids}
                for metric, prompt, response, token_ids, label in model_training_data_regular
            ],
            'mismatches': [
                {'word': get_word_from_probe(metric, word_to_probe), 'prompt': prompt, 'response': response, 'label': label, 'token_ids': token_ids}
                for metric, prompt, response, token_ids, label in model_training_data_mismatches
            ],
            'dontthinks': [
                {'word': get_word_from_probe(metric, word_to_probe), 'prompt': prompt, 'response': response, 'label': label, 'token_ids': token_ids}
                for metric, prompt, response, token_ids, label in ultrachat_training_data_dontthinks
            ],
            'twowords': [
                {'word1': get_word_from_probe(metric1, word_to_probe), 'word2': get_word_from_probe(metric2, word_to_probe),
                 'prompt': prompt, 'response': response, 'label1': label1, 'label2': label2, 'token_ids': token_ids}
                for metric1, metric2, prompt, response, token_ids, label1, label2 in model_training_data_twowords
            ]
        }
        
        with open(training_data_path, 'w') as f:
            json.dump(save_data, f, indent=2)

    (model, 
     step_to_obf_loss_dontthink_regular, step_to_obf_loss_regular,
     step_to_beh_loss_dontthink_regular, step_to_beh_loss_regular,
     step_to_obf_loss_twoword_dontthink_match, step_to_obf_loss_twoword_dontthink_mismatch,
     step_to_obf_loss_benign_dontthink_mismatch, step_to_obf_loss_dontthink_mismatch,
     step_to_layer_norm_dontthink_regular, step_to_layer_norm_regular,
     step_to_layer_norm_twoword_dontthink_match, step_to_layer_norm_twoword_dontthink_mismatch,
     step_to_layer_norm_benign_dontthink_mismatch, step_to_layer_norm_dontthink_mismatch) = train_with_gradient_context(
        model=model, 
        metric_in_out_metricgoal_regular=model_training_data_regular,
        metric_in_out_metricgoal_mismatches=model_training_data_mismatches,
        metric_in_out_metricgoal_benigndontthinks=ultrachat_training_data_dontthinks,
        metric2_in_out_metricgoal2_twowords=model_training_data_twowords,
        logger=logger, 
        lambda_=behaviour_lambda, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        target_layers=target_layers,
        mismatch_frequency=mismatch_frequency,
        benign_dontthink_frequency=dontthink_frequency,
        twoword_frequency=twoword_frequency
    )
    if not custom_folder_name:
        checkpoint_path = f"experiments/self_obfuscation_main/model_checkpoints/checkpoint_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_layer_{target_layers[0]}"
    else:
        checkpoint_path = f"experiments/self_obfuscation_main/model_checkpoints/{custom_folder_name}"
    
    # Optional step: save checkpoint...
    if save_model_checkpoint:
        logger.print(f"Saving model checkpoint to {checkpoint_path}...")
        model.model.save_pretrained(checkpoint_path)
    
    # Save model training metadata
    model_details["epoch"] = num_epochs
    model_details["step"] = len(model_training_data_regular) * num_epochs
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(checkpoint_path, "model_training_metadata.json"), "w") as f:
        json.dump(model_details, f, indent=4)
    plot_training_losses(
        step_to_obf_loss_dontthink_regular, step_to_obf_loss_regular,
        step_to_beh_loss_dontthink_regular, step_to_beh_loss_regular,
        step_to_obf_loss_twoword_dontthink_match, step_to_obf_loss_twoword_dontthink_mismatch,
        step_to_obf_loss_benign_dontthink_mismatch, step_to_obf_loss_dontthink_mismatch
    )
    
    plot_layer_norms(
        step_to_layer_norm_dontthink_regular, step_to_layer_norm_regular,
        step_to_layer_norm_twoword_dontthink_match, step_to_layer_norm_twoword_dontthink_mismatch,
        step_to_layer_norm_benign_dontthink_mismatch, step_to_layer_norm_dontthink_mismatch
    )

    logger.print(f"\nPOST-TRAINING TESTING...")
    for probed_word in training_words + test_words:
        if probed_word in word_to_probe:
            test_probe_on_model(
                model=model,
                metric=word_to_probe[probed_word], 
                logger=logger, 
                probed_word=probed_word, 
                test_words=[probed_word],
                pregen_data=pregen_data,
                model_details=model_details,
                layers_to_probe=target_layers,
                custom_folder_name=custom_folder_name
            )

    torch.cuda.empty_cache() # Fearful cleanup just in case
    return model, model_details, pregen_data

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and test conditional model')
    parser.add_argument('--model_name', type=str, default="gemma_2_9b_instruct",
                      help='Name of the model to use')
    parser.add_argument('--test_data_json', type=str, 
                      default="conditional_test_data_20250204_220618.json",
                      help='Name of test data JSON file')
    parser.add_argument('--probe_checkpoints', type=str,
                      default="20250205_002143_logistic_8",
                      help='Name of probe checkpoints directory')
    parser.add_argument('--ultrachat_training_json', type=str,
                      default="ultrachat_model_training_data_20250215_023915.json",
                      help='Name of ultrachat model training JSON file')
    parser.add_argument('--target_layer', type=int, default=8,
                      help='Target layer number')
    parser.add_argument('--num_epochs', type=int, default=5,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-6,
                      help='Learning rate for training')
    parser.add_argument('--behaviour_lambda', type=float, default=0.1,
                      help='Lambda parameter for behaviour loss')
    parser.add_argument('--num_sentences_per_word', type=int, default=20,
                      help='Number of sentences to generate per word for training')
    parser.add_argument('--num_test_sentences_per_word', type=int, default=10,
                      help='Number of sentences to generate per word for testing')
    parser.add_argument('--mismatch_frequency', type=float, default=0.0,
                      help='Frequency of mismatched data in training')
    parser.add_argument('--dontthink_frequency', type=float, default=0.2,
                      help='Frequency of benign clued data in training')
    parser.add_argument('--twoword_frequency', type=float, default=0.5,
                      help='Frequency of two-word data in training')
    parser.add_argument('--regenerate_training_data', type=bool, default=False,
                      help='Whether to use latest saved training data instead of generating new data')
    parser.add_argument('--training_data_json', type=str,
                      default="model_training_data.json",
                      help='JSON file to save/load training data')
    parser.add_argument('--custom_folder_name', type=str,
                      default="",
                      help='Custom folder name for probe scores')
    parser.add_argument('--do_pre_training_testing', action='store_true',
                      help='Whether to perform pre-training testing')
    parser.add_argument('--save_model_checkpoint', action='store_true',
                      help='Whether to save the model checkpoint after training')
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    train_and_test_conditional_model(
        model_name=args.model_name,
        test_data_json=args.test_data_json,
        probe_checkpoints=args.probe_checkpoints,
        ultrachat_training_json=args.ultrachat_training_json,
        target_layer=args.target_layer,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        behaviour_lambda=args.behaviour_lambda,
        num_sentences_per_word=args.num_sentences_per_word,
        num_test_sentences_per_word=args.num_test_sentences_per_word,
        mismatch_frequency=args.mismatch_frequency,
        dontthink_frequency=args.dontthink_frequency,
        twoword_frequency=args.twoword_frequency,
        regenerate_training_data=args.regenerate_training_data,
        training_data_json=args.training_data_json,
        custom_folder_name=args.custom_folder_name,
        do_pre_training_testing=args.do_pre_training_testing,
        save_model_checkpoint=args.save_model_checkpoint
    ) 