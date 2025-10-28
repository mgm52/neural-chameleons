import glob
import os
import json
import pickle
import random
from datetime import datetime
import argparse

import yaml
from datasets import load_dataset

from experiments.self_obfuscation_v0.stage1_probe_train import train_and_save_probes
from experiments.self_obfuscation_v0.utils_testing import test_probe_on_model
from obf_reps.logging import CSVTXTLogger
from experiments.self_obfuscation_v0.utils_misc import (
    VariedDirectPromptFormatter,
    cut_to_first_and_last_sentence,
    cut_to_first_sentence,
    load_local_model,
    run_script,
)
from experiments.self_obfuscation_v0.utils_training import words_data_to_probes


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train probes for conditional experiments")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma_2_9b_instruct",
        help="Model name to use (default: gemma_2_9b_instruct)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="experiments/self_obfuscation_main/outputs/model_checkpoints/20250228_053913_gemma_2_9b_instruct_plr_4e-05_l12_logistic_ptwf_0.5_ps_400_ms_200_mlr_2e-06_bl_0.1_good",
        help="Path to a fine-tuned model checkpoint to use instead of the base model",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="logistic",
        choices=["logistic", "mlp", "attention"],
        help="Type of probe to use (logistic, mlp, or attention)",
    )
    parser.add_argument(
        "--custom_folder_name",
        type=str,
        default="",
        help="Custom folder name for the probe checkpoints (default: '')",
    )
    parser.add_argument(
        "--train_test_probes",
        type=str,
        default="test_only",
        choices=["both", "train_only", "test_only"],
        help="Which probes to train (default: test_only)",
    )
    args = parser.parse_args()
    
    def load_config(config_path="config.yaml"):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_latest_file(pattern, directory=None):
        """
        Return the path to the most recently created file that matches the given
        pattern. If directory is provided, the pattern is joined with the directory.
        """
        if directory:
            pattern = os.path.join(directory, pattern)
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getctime)

    config = load_config("configs/self_obfuscation/config.yaml")
    general_settings = config["general_settings"]

    training_words = general_settings["training_words"]
    test_words = general_settings["test_words"]

    latest_probe_training_path = get_latest_file(
        # Hack: look for *instruct.json to differentiate from twoword.json...
        "probe_training_training_words_sentences_dict*instruct.json",
        "data/self_obfuscation/outputs"
    )
    if not latest_probe_training_path:
        raise ValueError("No probe training data found")
    latest_probe_testing_path = get_latest_file(
        # Hack: look for *instruct.json to differentiate from twoword.json...
        "probe_training_test_words_sentences_dict*instruct.json",
        "data/self_obfuscation/outputs"
    )
    if not latest_probe_testing_path:
        raise ValueError("No probe testing data found")

    probe_settings = config["probe_settings"]
    probe_benign_frequency = probe_settings["benign_frequency"]
    probe_learning_rate = probe_settings["learning_rate"]
    probe_batch_size = probe_settings["batch_size"]
    probe_num_epochs = probe_settings["num_epochs"]
    probe_twoword_frequency = probe_settings["twoword_frequency"]
    probe_num_sentences_per_word = probe_settings["num_sentences_per_word"]

    model_settings = config["model_settings"]
    target_layer = model_settings["target_layer"]

    ### New code for both training & testing

    # Load pre-generated test data
    print(f"Loading test data from data/self_obfuscation/outputs/conditional_test_data_gemma_2_9b_instruct_20250226_010745.json")
    test_data_path = "data/self_obfuscation/outputs/conditional_test_data_gemma_2_9b_instruct_20250226_010745.json"
    with open(test_data_path, "r") as f:
        pregen_data = json.load(f)

    logger = CSVTXTLogger(print_logs_to_console=False)

    model = load_local_model(checkpoint_path=args.model_checkpoint, model_name=args.model_name)
    metadata_path = os.path.join(args.model_checkpoint, "model_training_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            model_details = json.load(f)
        logger.print(f"Loaded model training metadata from checkpoint")

    for probe_num_epochs in [1, 2, 4]:
        for probe_learning_rate in [4e-5, 2e-5, 8e-5]:

            test_word_to_probe, test_probe_dir = train_and_save_probes(
                train_json_path=latest_probe_training_path,
                test_json_path=latest_probe_testing_path,
                training_words=training_words,
                test_words=test_words,
                model_name=args.model_name,
                model_checkpoint=args.model_checkpoint,
                model_in_memory=model,
                target_layers=target_layer,
                probe_type=args.probe_type,
                learning_rate=float(probe_learning_rate),
                batch_size=int(probe_batch_size),
                num_epochs=int(probe_num_epochs),
                twoword_frequency=float(probe_twoword_frequency),
                num_sentences_per_word=int(probe_num_sentences_per_word),
                custom_folder_name=args.custom_folder_name,
                benign_proportion_in_nonwords=float(probe_benign_frequency),
                train_test_probes="test_only",
                logger=logger
            )

            for probed_word in test_word_to_probe.keys():
                test_probe_on_model(
                    model=model,
                    metric=test_word_to_probe[probed_word],
                    logger=logger,
                    probed_word=probed_word,
                    test_words=[probed_word],
                    pregen_data=pregen_data,
                    model_details=model_details,
                    layers_to_probe=[int(target_layer)],
                    custom_folder_name=f"pne_{probe_num_epochs}_plr_{probe_learning_rate}",
                    generate_new_responses=False
                )
            
            # -----------------
            # STEP 4: Generate plots
            # -----------------
            latest_scores = get_latest_file(
                "probe_scores_*.csv",
                "experiments/self_obfuscation_main/outputs/results"
            )
            if not latest_scores:
                raise FileNotFoundError(
                    "No probe scores file found. Please ensure a probe-scoring step was run."
                )

            run_script(
                "experiments/self_obfuscation_main/stage4_massive_plotting.py",
                probe_scores_csv=latest_scores
            )