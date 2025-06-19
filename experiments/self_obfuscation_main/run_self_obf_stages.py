import gc
import glob
import json
import os
import sys
import subprocess
import yaml
from datetime import datetime

import torch

# Import the functions from the modules
from stage1_probe_train import train_and_save_probes
from stage2_model_train import train_and_test_conditional_model
from utils_testing import test_probe_on_model
from obf_reps.logging import CSVTXTLogger


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


def run_script(script_path, **kwargs):
    """
    Run a Python script located at script_path with the given named arguments.
    Each key-value pair in kwargs is passed as --key value on the command line.

    If the script returns a non-zero exit code, an error log is generated.
    """
    args = [sys.executable, script_path]
    for key, value in kwargs.items():
        if value is not None:
            args.extend([f"--{key}", str(value)])

    print(f"\nRunning {os.path.basename(script_path)}...")
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.returncode
    except subprocess.CalledProcessError as exc:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_msg = (
            f"Script {script_path} failed with exit code {exc.returncode}\n"
            f"Command: {' '.join(args)}\n"
            f"Output: {exc.stdout}\n"
            f"Error: {exc.stderr}\n"
        )

        log_file = f"experiments/self_obfuscation_main/error_logs/error_{timestamp}.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            f.write(f"Error occurred at {timestamp}\n")
            f.write(error_msg)

        raise RuntimeError(error_msg)


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config_path="config.yaml"):
    # Load configuration from YAML file
    config = load_config(config_path)
    
    # Extract settings from config
    # Test settings
    test_settings = config["test_settings"]
    num_test_sentences_per_word = test_settings["num_test_sentences_per_word"]
    mismatch_frequency = test_settings["mismatch_frequency"]
    dontthink_frequency = test_settings["dontthink_frequency"]
    
    # General settings
    general_settings = config["general_settings"]
    regenerate_training_data = general_settings["regenerate_training_data"]
    train_test_probes_before_model = general_settings["train_test_probes_before_model"]
    do_pre_training_testing = general_settings["do_pre_training_testing"]
    save_model_checkpoint = general_settings["save_model_checkpoint"]
    stages_to_run = general_settings["stages_to_run"]
    
    # Probe settings
    probe_settings = config["probe_settings"]
    probe_benign_frequency = probe_settings["benign_frequency"]
    probe_learning_rate = probe_settings["learning_rate"]
    probe_batch_size = probe_settings["batch_size"]
    probe_num_epochs = probe_settings["num_epochs"]
    probe_twoword_frequency = probe_settings["twoword_frequency"]
    probe_num_sentences_per_word = probe_settings["num_sentences_per_word"]
    
    # Model settings
    model_settings = config["model_settings"]
    model_num_sentences_per_word = model_settings["num_sentences_per_word"]
    target_layer = model_settings["target_layer"]
    num_epochs = model_settings["num_epochs"]
    learning_rate = model_settings["learning_rate"]
    twoword_frequency = model_settings["twoword_frequency"]
    
    # Experiment variations
    experiment_variations = config["experiment_variations"]
    model_names = experiment_variations["model_names"]
    behaviour_lambdas = experiment_variations["behaviour_lambdas"]
    probe_types = experiment_variations["probe_types"]

    logger = CSVTXTLogger(print_logs_to_console=False)
    first_iteration = True

    for model_name in model_names:
        for behaviour_lambda in behaviour_lambdas:
            for probe_type in probe_types:
                try:
                    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                    custom_folder_name = (
                        f"{current_datetime}_{model_name}_plr_{probe_learning_rate}"
                        f"_l{target_layer}_{probe_type}_ptwf_{probe_twoword_frequency}"
                        f"_ps_{probe_num_sentences_per_word}_ms_{model_num_sentences_per_word}"
                        f"_mlr_{learning_rate}_bl_{behaviour_lambda}"
                    )
                    if do_pre_training_testing:
                        custom_folder_name += "_withbasetest"
                    if train_test_probes_before_model:
                        custom_folder_name += "_pretrainprobes"

                    # -----------------
                    # STEP 0: (Optional) Generate test data
                    # -----------------
                    if 0 in stages_to_run:
                        run_script(
                            "datasets/self_obfuscation/stage0_data_gen_synthetic.py",
                            model_name=model_name
                        )

                    # Get the latest test data (required for next steps)
                    latest_test_data = os.path.basename(
                        get_latest_file(
                            f"conditional_test_data_{model_name}*.json",
                            "experiments/self_obfuscation_main/results"
                        )
                    )
                    if not latest_test_data:
                        raise FileNotFoundError(
                            "No test data file found. Please run step 0 first."
                        )

                    # -----------------
                    # STEP 1: Train probes
                    # -----------------
                    if 1 in stages_to_run:
                        print("\nTraining probes...")
                        word_to_probe, probe_dir = train_and_save_probes(
                            model_name=model_name,
                            target_layers=target_layer,
                            probe_type=probe_type,
                            learning_rate=probe_learning_rate,
                            batch_size=probe_batch_size,
                            num_epochs=probe_num_epochs,
                            twoword_frequency=probe_twoword_frequency,
                            regenerate_training_data=regenerate_training_data,
                            num_sentences_per_word=probe_num_sentences_per_word,
                            custom_folder_name=custom_folder_name,
                            benign_proportion_in_nonwords=probe_benign_frequency,
                            logger=logger,
                            train_test_probes=(
                                "train_only" if not train_test_probes_before_model else "both"
                            )
                        )
                        latest_probe_dir = custom_folder_name
                    else:
                        latest_probe_dir = os.path.basename(
                            get_latest_file("*", "experiments/self_obfuscation_main/probe_checkpoints")
                        )
                        if not latest_probe_dir:
                            raise FileNotFoundError(
                                "No probe checkpoint directory found. "
                                "Please ensure step 1 completed successfully."
                            )
                        word_to_probe = None

                    # -----------------
                    # STEP 2: (Optional) Evaluate model with probes on raw chat data
                    # -----------------
                    if 2 in stages_to_run:
                        run_script(
                            "experiments/self_obfuscation_main/stage3_test_ultrachat.py",
                            model_checkpoint="none",  # Use the base model
                            probe_checkpoints=latest_probe_dir,
                            num_conversations=100,
                            model_name=model_name
                        )

                    # -----------------
                    # STEP 3: Train the model (finetune) using the probes
                    # -----------------
                    if 3 in stages_to_run:
                        latest_ultrachat = get_latest_file(
                            "ultrachat_model_training_data*.json",
                            "experiments/self_obfuscation_main/results"
                        )
                        if not latest_ultrachat:
                            raise FileNotFoundError(
                                "No ultrachat training data file found. "
                                "Please ensure previous steps completed successfully."
                            )

                        latest_training_data = get_latest_file(
                            f"model_training_data_{model_name}*.json",
                            "experiments/self_obfuscation_main/results"
                        )
                        # If existing training data is found and we are not regenerating,
                        # use that; otherwise create a new file name.
                        if latest_training_data and not regenerate_training_data:
                            training_data_json = os.path.basename(latest_training_data)
                        else:
                            training_data_json = (
                                f"model_training_data_{model_name}_"
                                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            )

                        print("\nTraining and testing conditional model...")
                        model, model_details, pregen_data = train_and_test_conditional_model(
                            model_name=model_name,
                            test_data_json=latest_test_data,
                            probe_checkpoints=latest_probe_dir,
                            ultrachat_training_json=os.path.basename(latest_ultrachat),
                            target_layer=int(target_layer),
                            num_epochs=num_epochs,
                            learning_rate=learning_rate,
                            behaviour_lambda=behaviour_lambda,
                            num_sentences_per_word=model_num_sentences_per_word,
                            num_test_sentences_per_word=num_test_sentences_per_word,
                            mismatch_frequency=mismatch_frequency,
                            dontthink_frequency=dontthink_frequency,
                            twoword_frequency=twoword_frequency,
                            regenerate_training_data=regenerate_training_data,
                            training_data_json=training_data_json,
                            custom_folder_name=custom_folder_name,
                            logger=logger,
                            word_to_probe=word_to_probe,
                            do_pre_training_testing=do_pre_training_testing,
                            save_model_checkpoint=save_model_checkpoint
                        )
                    else:
                        model, model_details, pregen_data = None, None, None

                    latest_model_checkpoint = custom_folder_name

                    # If we didn't train the test probes before the model finetune,
                    # train them now on the finetuned model.
                    if 3 in stages_to_run and latest_model_checkpoint:
                        if not train_test_probes_before_model:
                            print("\nTraining test probes on fine-tuned model...")
                            test_word_to_probe, test_probe_dir = train_and_save_probes(
                                model_name=model_name,
                                model_in_memory=model,
                                model_checkpoint=None,
                                target_layers=target_layer,
                                probe_type=probe_type,
                                learning_rate=probe_learning_rate,
                                batch_size=probe_batch_size,
                                num_epochs=probe_num_epochs,
                                twoword_frequency=probe_twoword_frequency,
                                regenerate_training_data=regenerate_training_data,
                                num_sentences_per_word=probe_num_sentences_per_word,
                                custom_folder_name=f"{custom_folder_name}_finetuned",
                                train_test_probes="test_only",
                                benign_proportion_in_nonwords=probe_benign_frequency,
                                logger=logger
                            )
                        else:
                            test_word_to_probe = word_to_probe
                            test_probe_dir = latest_probe_dir
                    else:
                        test_word_to_probe = None

                    logger.print("\nPOST-TRAINING TESTING...")

                    # If not already tested with probes before model training, do it here
                    if (not train_test_probes_before_model
                            and 3 in stages_to_run
                            and test_word_to_probe is not None
                            and model is not None):
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
                                custom_folder_name=custom_folder_name
                            )

                    # -----------------
                    # STEP 4: Generate plots
                    # -----------------
                    latest_scores = get_latest_file(
                        "probe_scores_*.csv",
                        "experiments/self_obfuscation_main/results"
                    )
                    if not latest_scores:
                        raise FileNotFoundError(
                            "No probe scores file found. Please ensure a probe-scoring step was run."
                        )

                    if 4 in stages_to_run:
                        run_script(
                            "experiments/self_obfuscation_main/stage4_massive_plotting.py",
                            probe_scores_csv=latest_scores
                        )

                    print("\nAll scripts completed successfully!")
                    gc.collect()
                    torch.cuda.empty_cache()
                    first_iteration = False

                except Exception as e:
                    first_iteration = False
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_file = f"experiments/self_obfuscation_main/error_logs/error_{timestamp}.log"
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)

                    with open(log_file, "w") as f:
                        f.write(f"Error occurred at {timestamp}\n")
                        f.write("Unhandled exception details:\n")
                        f.write(f"  model_name: {model_name}\n")
                        f.write(f"  behaviour_lambda: {behaviour_lambda}\n")
                        f.write(f"  probe_type: {probe_type}\n")
                        f.write(f"Exception message: {str(e)}\n")

                    print(f"\nError in iteration (see {log_file} for details). "
                          "Continuing with next iteration...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run self-obfuscation experiment stages")
    parser.add_argument("--config", type=str, default="configs/self_obfuscation/config.yaml", 
                        help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    main(config_path=args.config)
