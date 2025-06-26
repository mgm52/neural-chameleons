"""
Trains a harmful-jailbreak probe to distinguish between harmful and benign examples.

INPUTS:
- Model checkpoint or model name
- HuggingFace dataset: 'Mechanistic-Anomaly-Detection/gemma2-jailbreaks'

OUTPUTS:
- Trained probe (.pkl file) saved in probe_checkpoints directory
- Training metadata (JSON file) with configuration and dataset sizes
- Console logs with training progress and final checkpoint directory path
"""

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, model_validator
from utils_plotting import compute_metric_bootstrap

import wandb
from datasets import load_dataset
from experiments.self_obfuscation_main.utils_misc import load_local_model
from experiments.self_obfuscation_main.utils_testing import (
    test_trainable_metric_on_response_data_module,
)
from experiments.self_obfuscation_main.utils_training import ResponseProbeDataModule
from obf_reps.metrics import LogisticRegressionMetric, MetricConfig

harmful_dataset_names = [
    "circuit_breakers_train",
    "circuit_breakers_test",
    "harmful_autodan",
    "harmful_human_mt",
    "harmful_gcg",
    "harmful_msj",
    "harmful_pair",
    "harmful_instructions_train",
    "harmful_instructions_test",
    "harmful_prefill",
]

benign_dataset_names = [
    "xstest",
    "benign_instructions_train",
    "benign_instructions_test",
]


class ProbeTrainingConfig(BaseModel):
    model_name_or_path: str | Path
    learning_rate: float = 4e-5
    batch_size: int = 64
    num_epochs: int = 1
    target_layers: list[int] = [12]
    probe_types: str | list[str] = "logistic"
    training_datasets: dict[str, dict[str, list[str]]]
    test_datasets: dict[str, dict[str, list[str]]] | None = None
    # WandB settings
    wandb_project: str = "probe-training"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None

    @model_validator(mode="after")
    def validate_probe_types(self):
        if isinstance(self.probe_types, str):
            self.probe_types = [self.probe_types]

            # Validate training datasets
            for split_name, split_config in self.training_datasets.items():
                assert isinstance(
                    split_config, dict
                ), f"Training dataset {split_name} must be a dict with 'harmful' and 'benign' keys"
                assert set(split_config.keys()) <= {
                    "harmful",
                    "benign",
                }, f"Training dataset {split_name} can only have 'harmful' and 'benign' keys, got: {list(split_config.keys())}"

                for dataset_type, datasets in split_config.items():
                    if dataset_type == "harmful":
                        assert all(
                            dataset in harmful_dataset_names for dataset in datasets
                        ), f"Invalid harmful dataset in {split_name}: {datasets}. Valid datasets: {harmful_dataset_names}"
                    elif dataset_type == "benign":
                        assert all(
                            dataset in benign_dataset_names for dataset in datasets
                        ), f"Invalid benign dataset in {split_name}: {datasets}. Valid datasets: {benign_dataset_names}"

        # Validate test datasets if provided
        if self.test_datasets is not None:
            for split_name, split_config in self.test_datasets.items():
                assert isinstance(
                    split_config, dict
                ), f"Test dataset {split_name} must be a dict with 'harmful' and 'benign' keys"
                assert set(split_config.keys()) <= {
                    "harmful",
                    "benign",
                }, f"Test dataset {split_name} can only have 'harmful' and 'benign' keys, got: {list(split_config.keys())}"

                for dataset_type, datasets in split_config.items():
                    if dataset_type == "harmful":
                        assert all(
                            dataset in harmful_dataset_names for dataset in datasets
                        ), f"Invalid harmful dataset in {split_name}: {datasets}. Valid datasets: {harmful_dataset_names}"
                    elif dataset_type == "benign":
                        assert all(
                            dataset in benign_dataset_names for dataset in datasets
                        ), f"Invalid benign dataset in {split_name}: {datasets}. Valid datasets: {benign_dataset_names}"

        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path | str):
        assert os.path.exists(yaml_path), f"Config file {yaml_path} does not exist"
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(**config)


class WandBLogger:
    """WandB logger that maintains compatibility with CSVTXTLogger interface."""

    def __init__(self, config: ProbeTrainingConfig, print_logs_to_console: bool = True):
        self.print_logs_to_console = print_logs_to_console
        self.config = config
        self.run = None
        self.current_probe_type = None

    def initialize_run(self, probe_type: str, metadata: Dict):
        """Initialize a new WandB run for a specific probe type."""
        if self.run is not None:
            self.run.finish()

        self.current_probe_type = probe_type
        run_name = f"{probe_type}_probe"
        if self.config.wandb_name:
            run_name = f"{self.config.wandb_name}_{run_name}"

        tags = self.config.wandb_tags or []
        tags.append(probe_type)

        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            tags=tags,
            config={
                "model_name_or_path": str(self.config.model_name_or_path),
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "target_layers": self.config.target_layers,
                "probe_type": probe_type,
                **metadata,
            },
        )

    def print(self, msg: str) -> None:
        """Print message to console and log to WandB."""
        if self.print_logs_to_console:
            print(msg)
        if self.run:
            self.run.log({"console_output": msg})

    def optional_print(self, msg: str) -> None:
        """Print message to console if enabled."""
        if self.print_logs_to_console:
            print(msg)

    def log(self, data: Dict[str, Union[int, float, str]]) -> None:
        """Log key-value pairs to WandB."""
        if self.run:
            self.run.log(data)

    def log_to_table(self, data: List, table_name: str) -> None:
        """Log tabular data to WandB."""
        if self.run:
            table = wandb.Table(
                data=data,
                columns=[f"col_{i}" for i in range(len(data[0]))] if data else [],
            )
            self.run.log({table_name: table})

    def create_table(self, table_name: str, columns: List[str]) -> None:
        """Create empty table (no-op for WandB)."""
        pass

    def log_tables(self) -> None:
        """No-op for WandB compatibility."""
        pass

    def log_table_name(self, table_name: str) -> None:
        """No-op for WandB compatibility."""
        pass

    def log_metrics(self, metrics: Dict[str, Dict[str, float]], probe_type: str):
        """Log evaluation metrics to WandB."""
        flattened_metrics = {}
        for split_name, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                key = f"{split_name}/{metric_name}"
                flattened_metrics[key] = value

        if self.run:
            self.run.log(flattened_metrics)

    def log_scores(self, scores: Dict[str, Dict[str, List[float]]], probe_type: str):
        """Log raw scores as histograms to WandB."""
        if not self.run:
            return

        for split_name, split_scores in scores.items():
            for score_type, score_list in split_scores.items():
                if score_list:  # Only log if there are scores
                    self.run.log(
                        {
                            f"{split_name}/{score_type}_histogram": wandb.Histogram(score_list),
                            f"{split_name}/{score_type}_mean": sum(score_list) / len(score_list),
                            f"{split_name}/{score_type}_std": (
                                sum(
                                    (x - sum(score_list) / len(score_list)) ** 2
                                    for x in score_list
                                )
                                / len(score_list)
                            )
                            ** 0.5,
                        }
                    )

    def finish(self):
        """Finish the current WandB run."""
        if self.run:
            self.run.finish()
            self.run = None


def train_harmful_probe(
    model,
    harmful_examples,
    benign_examples,
    logger,
    target_layers,
    probe_type="logistic",
    learning_rate=5e-4,
    batch_size=64,
    num_epochs=5,
):
    """
    Train a probe to distinguish between harmful and benign examples.

    Args:
        model: The model to extract representations from
        harmful_examples: List of (prompt, response) tuples for harmful examples
        benign_examples: List of (prompt, response) tuples for benign examples
        logger: Logger for tracking training progress
        target_layers: List of layer indices to probe
        probe_type: Type of probe to use ("logistic" or "mlp")
        learning_rate: Learning rate for probe training
        batch_size: Batch size for probe training
        num_epochs: Number of epochs for probe training

    Returns:
        Trained probe
    """
    logger.print(
        f"Creating data module with {len(harmful_examples)} harmful and {len(benign_examples)} benign examples..."
    )

    # Create the data module using ResponseProbeDataModule from conditional_training.py
    data_module = ResponseProbeDataModule(
        topical_prompts_and_responses=harmful_examples,
        nontopical_prompts_and_responses=benign_examples,
        batch_size=batch_size,
    )

    logger.print("Configuring probe...")
    # Configure which layer(s) to probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Create the probe
    if probe_type == "logistic":
        metric = LogisticRegressionMetric(
            model=model, data_module=data_module, config=config, logger=logger
        )
    else:
        from obf_reps.metrics import MLPMetric

        metric = MLPMetric(model=model, data_module=data_module, config=config, logger=logger)

    logger.print("Finished training probe")
    # Freeze probe
    metric.probe[0].requires_grad_(False)

    return metric


def preprocess_dataset(dataset):
    clean_dataset = []
    for x in dataset:
        prompt = str(x[0])
        response = str(x[1])

        # Remove special tokens
        prompt = prompt.replace("<bos><start_of_turn>user\n", "")
        prompt = prompt.replace("<end_of_turn>\n<start_of_turn>model\n", "")

        clean_dataset.append((prompt, response))

    return clean_dataset


def extract_examples_from_datasets(dataset_names, jailbreaks_dataset):
    """Extract examples from multiple datasets without sampling."""
    examples = []

    for dataset_name in dataset_names:
        if dataset_name in jailbreaks_dataset:
            dataset = jailbreaks_dataset[dataset_name]
            # Extract (prompt, completion) tuples from each dataset
            for item in dataset:
                prompt = item["prompt"]
                completion = item.get("completion", "")
                examples.append((prompt, completion))

    return examples


def load_datasets_from_config(config, jailbreaks_dataset):
    """Load datasets based on configuration."""
    datasets = {}

    # Load training datasets
    for split_name, split_config in config.training_datasets.items():
        datasets[split_name] = {}
        for dataset_type, dataset_names in split_config.items():
            # Extract all examples from the specified datasets
            raw_examples = extract_examples_from_datasets(dataset_names, jailbreaks_dataset)
            # Preprocess the examples
            datasets[split_name][dataset_type] = preprocess_dataset(raw_examples)

    # Load test datasets if provided
    if config.test_datasets:
        for split_name, split_config in config.test_datasets.items():
            datasets[split_name] = {}
            for dataset_type, dataset_names in split_config.items():
                # Extract all examples from the specified datasets
                raw_examples = extract_examples_from_datasets(dataset_names, jailbreaks_dataset)
                # Preprocess the examples
                datasets[split_name][dataset_type] = preprocess_dataset(raw_examples)

    return datasets


def train_and_evaluate_probes(config):
    """Train and evaluate multiple probes with different configurations."""
    # Set up logger
    logger = WandBLogger(config, print_logs_to_console=True)
    logger.print("Starting probe training and evaluation...")

    # Load jailbreaks dataset
    jailbreaks_dataset = load_dataset("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")

    # Load datasets based on configuration
    datasets = load_datasets_from_config(config, jailbreaks_dataset)

    # Print dataset information
    for split_name, split_data in datasets.items():
        logger.print(f"{split_name.upper()} datasets:")
        for dataset_type, examples in split_data.items():
            logger.print(f"  {dataset_type}: {len(examples)} examples")

    # Load model for probe training
    logger.print("Loading model for probe training...")
    if os.path.isdir(str(config.model_name_or_path)):
        logger.print(f"Loading model from checkpoint directory: {config.model_name_or_path}")
        model = load_local_model(checkpoint_path=str(config.model_name_or_path))
    else:
        logger.print(f"Loading model by name: {config.model_name_or_path}")
        model = load_local_model(model_name=str(config.model_name_or_path))

    # Results storage
    all_results = {}

    # Train each probe type
    for probe_type in config.probe_types:
        logger.print(f"\n=== Training {probe_type} probe ===")

        # Create directory for this probe
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = (
            f"self_obfuscation_experiment/outputs/probe_checkpoints/{date_time}_{probe_type}_probe"
        )
        os.makedirs(directory, exist_ok=True)

        # Save metadata
        metadata = {
            "date_time": date_time,
            "target_layers": config.target_layers,
            "probe_type": probe_type,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "model_name_or_path": str(config.model_name_or_path),
            "dataset_sizes": {
                split: {dtype: len(data) for dtype, data in split_data.items()}
                for split, split_data in datasets.items()
            },
            "directory": directory,
        }

        # Initialize WandB run for this probe type
        logger.initialize_run(probe_type, metadata)

        # Train the probe
        trained_probe = train_harmful_probe(
            model=model,
            harmful_examples=datasets["train"]["harmful"],
            benign_examples=datasets["train"]["benign"],
            logger=logger,
            target_layers=config.target_layers,
            probe_type=probe_type,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
        )

        with open(f"{directory}/metadata.json", "w") as f:
            import json

            json.dump(metadata, f, indent=2)

        # Save the probe (remove model reference first)
        trained_probe.model = None
        with open(f"{directory}/probe.pkl", "wb") as f:
            pickle.dump(trained_probe, f)

        logger.print(f"Saved {probe_type} probe to {directory}/probe.pkl")

        # Restore model reference for evaluation
        trained_probe.model = model

        # Evaluate on all available datasets
        probe_results = {}
        all_scores = {}

        for split_name, split_data in datasets.items():
            if "harmful" in split_data and "benign" in split_data:
                logger.print(f"Evaluating {probe_type} probe on {split_name} set...")

                # Create data module for evaluation
                eval_data_module = ResponseProbeDataModule(
                    topical_prompts_and_responses=split_data["harmful"],
                    nontopical_prompts_and_responses=split_data["benign"],
                    batch_size=config.batch_size,
                )

                # Get scores
                scores = test_trainable_metric_on_response_data_module(
                    model,
                    trained_probe,
                    logger,
                    eval_data_module,
                    metadata,
                    config.target_layers,
                    f"{probe_type}_probe_{split_name}",
                )

                pos_scores = [score["pos_score"] for score in scores]
                neg_scores = [score["neg_score"] for score in scores]

                # Store raw scores
                all_scores[split_name] = {
                    "pos_scores": pos_scores,
                    "neg_scores": neg_scores,
                }

                # Compute metrics
                recall1fpr = compute_metric_bootstrap(
                    "recall@1",
                    positive_scores=pos_scores,
                    negative_scores=neg_scores,
                    confidence_level=0.95,
                )
                recall01fpr = compute_metric_bootstrap(
                    "recall@0.01",
                    positive_scores=pos_scores,
                    negative_scores=neg_scores,
                    confidence_level=0.95,
                )

                probe_results[split_name] = {
                    "recall@1fpr": recall1fpr,
                    "recall@0.01fpr": recall01fpr,
                }

                logger.print(f"  Recall@1%FPR: {recall1fpr}")
                logger.print(f"  Recall@0.01%FPR: {recall01fpr}")

        # Log metrics and scores to WandB
        logger.log_metrics(probe_results, probe_type)
        logger.log_scores(all_scores, probe_type)

        # Save detailed results
        with open(f"{directory}/results.json", "w") as f:
            json.dump({"metrics": probe_results, "scores": all_scores}, f, indent=2)

        # Save summary CSV
        with open(f"{directory}/summary.csv", "w") as f:
            f.write("split,metric,value\n")
            for split_name, metrics in probe_results.items():
                for metric_name, value in metrics.items():
                    f.write(f"{split_name},{metric_name},{value}\n")

        logger.print(f"Saved detailed results to {directory}/")

        # Store in overall results
        all_results[probe_type] = {
            "directory": directory,
            "metrics": probe_results,
            "scores": all_scores,
        }

    # Finish WandB logging
    logger.finish()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = ProbeTrainingConfig.from_yaml(args.config)
    results = train_and_evaluate_probes(config)

    # Print final summary
    print("\n=== FINAL SUMMARY ===")
    for probe_type, probe_results in results.items():
        print(f"\n{probe_type.upper()} PROBE:")
        print(f"  Directory: {probe_results['directory']}")
        for split_name, metrics in probe_results["metrics"].items():
            print(f"  {split_name}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value[0]:.3f} ({value[1]:.3f}-{value[2]:.3f})")
