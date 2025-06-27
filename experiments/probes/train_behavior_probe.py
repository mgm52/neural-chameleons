"""
Trains a behavioral probe to distinguish between positive and negative examples.

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
from typing import Dict, List, Literal, Optional, Union

import yaml
from dataset_adapters import get_adapter
from pydantic import BaseModel, model_validator
from utils_plotting import compute_metric_bootstrap

import wandb
from experiments.self_obfuscation_main.utils_misc import load_local_model
from experiments.self_obfuscation_main.utils_testing import (
    test_trainable_metric_on_response_data_module,
)
from experiments.self_obfuscation_main.utils_training import ResponseProbeDataModule
from obf_reps.logging import Logger
from obf_reps.metrics import (
    LogisticRegressionMetric,
    MetricConfig,
    MLPMetric,
    SklearnLogisticRegressionMetric,
)

positive_dataset_names = [
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
    "wildjailbreak_harmful",
]

negative_dataset_names = [
    "xstest",
    "benign_instructions_train",
    "benign_instructions_test",
    "wildjailbreak_benign",
]


class ProbeConfig(BaseModel):
    type: Literal["logistic", "mlp", "sklearn_logistic"]
    pooling_type: Literal["mean", "max", "attn", "ema"]
    batch_size: int
    num_epochs: int
    target_layers: list[int]
    learning_rate: float | None = None
    C: float | None = None

    @model_validator(mode="after")
    def validate_config(self):
        if self.type == "sklearn_logistic":
            assert self.C is not None, "C must be provided for sklearn_logistic probe"
        if self.type in ["logistic", "mlp"]:
            assert (
                self.learning_rate is not None
            ), "learning_rate must be provided for logistic and mlp probes"
        return self


class ProbeTrainingConfig(BaseModel):
    model_name_or_path: str | Path
    probes: dict[str, ProbeConfig]  # Dictionary of probe configs
    training_datasets: dict[str, dict[str, list[str]]]
    test_datasets: dict[str, dict[str, list[str]]] | None = None
    wandb_project: str = "probe-training"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[list[str]] = None

    @model_validator(mode="after")
    def validate_config(self):
        # Validate probes
        assert self.probes, "At least one probe must be specified"
        for probe_name, probe_config in self.probes.items():
            assert probe_config.type in ["logistic", "mlp", "sklearn_logistic"], (
                f"Invalid probe type '{probe_config.type}' for probe '{probe_name}'. "
                "Valid types are: 'logistic', 'mlp', 'sklearn_logistic'"
            )

        # Validate training datasets
        for split_name, split_config in self.training_datasets.items():
            assert isinstance(
                split_config, dict
            ), f"Training dataset {split_name} must be a dict with 'positive' and 'negative' keys"
            assert set(split_config.keys()) <= {
                "positive",
                "negative",
            }, f"Training dataset {split_name} can only have 'positive' and 'negative' keys, got: {list(split_config.keys())}"

            for dataset_type, datasets in split_config.items():
                if dataset_type == "positive":
                    assert all(
                        dataset in positive_dataset_names for dataset in datasets
                    ), f"Invalid positive dataset in {split_name}: {datasets}. Valid datasets: {positive_dataset_names}"
                elif dataset_type == "negative":
                    assert all(
                        dataset in negative_dataset_names for dataset in datasets
                    ), f"Invalid negative dataset in {split_name}: {datasets}. Valid datasets: {negative_dataset_names}"

        # Validate test datasets if provided
        if self.test_datasets is not None:
            for split_name, split_config in self.test_datasets.items():
                assert isinstance(
                    split_config, dict
                ), f"Test dataset {split_name} must be a dict with 'positive' and 'negative' keys"
                assert set(split_config.keys()) <= {
                    "positive",
                    "negative",
                }, f"Test dataset {split_name} can only have 'positive' and 'negative' keys, got: {list(split_config.keys())}"

                for dataset_type, datasets in split_config.items():
                    if dataset_type == "positive":
                        assert all(
                            dataset in positive_dataset_names for dataset in datasets
                        ), f"Invalid positive dataset in {split_name}: {datasets}. Valid datasets: {positive_dataset_names}"
                    elif dataset_type == "negative":
                        assert all(
                            dataset in negative_dataset_names for dataset in datasets
                        ), f"Invalid negative dataset in {split_name}: {datasets}. Valid datasets: {negative_dataset_names}"

        return self

    @classmethod
    def from_yaml(cls, yaml_path: Path | str):
        assert os.path.exists(yaml_path), f"Config file {yaml_path} does not exist"
        with open(yaml_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return cls(**config)


class WandBLogger(Logger):
    """WandB logger that maintains compatibility with obf_reps Logger interface."""

    def __init__(
        self,
        config: ProbeTrainingConfig,
        log_file: Optional[str] = None,
        username: Optional[str] = None,
        metadata: Optional[Dict] = None,
        print_logs_to_console: bool = True,
    ):
        self.print_logs_to_console = print_logs_to_console
        self.config = config
        self.run = None
        self.current_probe_type = None
        self.tables: Dict[str, wandb.Table] = {}

    def print(self, msg: str) -> None:
        """Print message to console and log to WandB."""
        if self.print_logs_to_console:
            print(msg)
        if self.run:
            self.run.log({"console_output": msg})

    def log(self, data: Dict[str, Union[int, float, str]]) -> None:
        """Log key-value pairs to WandB."""
        if self.run:
            self.run.log(data)

    def log_to_table(self, data: List, table_name: str) -> None:
        """Log tabular data to WandB."""
        if table_name in self.tables and self.run:
            # Add data to existing table
            num_columns = len(self.tables[table_name].columns)
            if len(data) != num_columns:
                raise ValueError(
                    f"Data length {len(data)} does not match table size {num_columns}"
                )
            self.tables[table_name].add_data(*data)

    def create_table(self, table_name: str, columns: List[str]) -> None:
        """Create a new WandB table."""
        self.tables[table_name] = wandb.Table(columns=columns)

    def log_tables(self) -> None:
        """Log all tables to WandB."""
        if self.run:
            self.run.log(self.tables)

    def log_table_name(self, table_name: str) -> None:
        """Log a specific table to WandB."""
        if self.run and table_name in self.tables:
            table = wandb.Table(
                columns=self.tables[table_name].columns,
                data=self.tables[table_name].data,
            )
            self.run.log({table_name: table})

    def __del__(self):
        """Finish the current WandB run and log tables."""
        if self.run:
            self.log_tables()
            self.run.finish()

    # Additional helper methods for probe training
    def initialize_run(self, probe_name: str, metadata: Dict):
        """Initialize a new WandB run for a specific probe configuration."""
        if self.run is not None:
            self.run.finish()

        self.current_probe_type = metadata.get("probe_type", "unknown")
        run_name = probe_name
        if self.config.wandb_name:
            run_name = f"{self.config.wandb_name}_{run_name}"

        tags = self.config.wandb_tags or []
        tags.extend([probe_name, metadata.get("probe_type", "unknown")])

        self.run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            tags=tags,
            config={
                "model_name_or_path": str(self.config.model_name_or_path),
                **metadata,
            },
        )

    def optional_print(self, msg: str) -> None:
        """Print message to console if enabled."""
        if self.print_logs_to_console:
            print(msg)

    def log_metrics(self, metrics: Dict[str, Dict[str, float]], probe_name: str):
        """Log evaluation metrics to WandB as separate tables for each metric."""
        if not self.run:
            return

        # Create separate table for each metric type
        for split_name, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                table_name = f"{probe_name}_{metric_name}"

                # Create table if it doesn't exist
                if table_name not in self.tables:
                    self.create_table(table_name, ["split", "value", "ci_lower", "ci_upper"])

                if isinstance(value, (list, tuple)) and len(value) == 3:
                    # Value is [estimate, ci_lower, ci_upper]
                    self.log_to_table(
                        [split_name, value[0], value[1], value[2]],
                        table_name,
                    )
                else:
                    # Single value
                    self.log_to_table([split_name, value, None, None], table_name)

                # Log the table immediately
                self.log_table_name(table_name)

    def log_scores(self, scores: Dict[str, Dict[str, List[float]]], probe_name: str):
        """Log score statistics to WandB as separate tables for each score type."""
        if not self.run:
            return

        for split_name, split_scores in scores.items():
            for score_type, score_list in split_scores.items():
                if score_list:  # Only log if there are scores
                    table_name = f"{probe_name}_{score_type}_scores"

                    # Create table if it doesn't exist
                    if table_name not in self.tables:
                        self.create_table(table_name, ["split", "mean", "std", "count"])

                    mean_score = sum(score_list) / len(score_list)
                    std_score = (
                        sum((x - mean_score) ** 2 for x in score_list) / len(score_list)
                    ) ** 0.5

                    self.log_to_table(
                        [split_name, mean_score, std_score, len(score_list)],
                        table_name,
                    )

                    # Log the table immediately
                    self.log_table_name(table_name)

    def log_combined_results(
        self,
        metrics: Dict[str, Dict[str, float]],
        scores: Dict[str, Dict[str, List[float]]],
        probe_name: str,
    ):
        """Deprecated: No longer creates combined table since we use separate tables for each metric."""
        pass

    def finish(self):
        """Finish the current WandB run."""
        if self.run:
            self.run.finish()
            self.run = None


def train_behavioral_probe(
    model,
    positive_examples,
    negative_examples,
    logger,
    target_layers,
    probe_type="logistic",
    learning_rate=5e-4,
    batch_size=64,
    num_epochs=5,
):
    """
    Train a probe to distinguish between positive and negative examples.

    Args:
        model: The model to extract representations from
        positive_examples: List of (prompt, response) tuples for positive examples
        negative_examples: List of (prompt, response) tuples for negative examples
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
        f"Creating data module with {len(positive_examples)} positive and {len(negative_examples)} negative examples..."
    )

    # Create the data module using ResponseProbeDataModule from conditional_training.py
    data_module = ResponseProbeDataModule(
        topical_prompts_and_responses=positive_examples,
        nontopical_prompts_and_responses=negative_examples,
        batch_size=batch_size,
    )

    # Configure which layer(s) to probe
    config = MetricConfig(
        layers=target_layers,
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )

    # Create the probe
    match probe_type:
        case "logistic":
            metric = LogisticRegressionMetric(
                model=model, data_module=data_module, config=config, logger=logger
            )
        case "mlp":
            metric = MLPMetric(model=model, data_module=data_module, config=config, logger=logger)
        case "sklearn_logistic":
            metric = SklearnLogisticRegressionMetric(
                model=model, data_module=data_module, config=config, logger=logger
            )
        case _:
            raise ValueError(f"Invalid probe type: {probe_type}")

    logger.print("Finished training probe")
    # Freeze probe
    if probe_type in ["logistic", "mlp"]:
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


def extract_examples_from_datasets(dataset_names):
    """Extract examples from multiple datasets using appropriate adapters."""
    examples = []

    for dataset_name in dataset_names:
        adapter = get_adapter(dataset_name)
        dataset_examples = adapter.load_and_extract_examples(dataset_name)
        examples.extend(dataset_examples)

    return examples


def load_datasets_from_config(config):
    """Load datasets based on configuration."""
    datasets = {}

    # Load training datasets
    for split_name, split_config in config.training_datasets.items():
        datasets[split_name] = {}
        for dataset_type, dataset_names in split_config.items():
            raw_examples = extract_examples_from_datasets(dataset_names)
            datasets[split_name][dataset_type] = preprocess_dataset(raw_examples)

    # Load test datasets if provided
    if config.test_datasets:
        for split_name, split_config in config.test_datasets.items():
            datasets[split_name] = {}
            for dataset_type, dataset_names in split_config.items():
                raw_examples = extract_examples_from_datasets(dataset_names)
                datasets[split_name][dataset_type] = preprocess_dataset(raw_examples)

    return datasets


def train_and_evaluate_probes(config):
    """Train and evaluate multiple probes with different configurations."""
    # Set up logger
    logger = WandBLogger(config, print_logs_to_console=True)
    logger.print("Starting probe training and evaluation...")

    # Load datasets based on configuration
    datasets = load_datasets_from_config(config)

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

    # Train each probe configuration
    for probe_name, probe_config in config.probes.items():
        logger.print(f"\n=== Training {probe_name} ({probe_config.type} probe) ===")

        # Get probe-specific parameters
        learning_rate = probe_config.learning_rate
        batch_size = probe_config.batch_size
        num_epochs = probe_config.num_epochs
        target_layers = probe_config.target_layers

        # Create directory for this probe
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = (
            f"self_obfuscation_experiment/outputs/probe_checkpoints/{date_time}_{probe_name}"
        )
        os.makedirs(directory, exist_ok=True)

        # Save metadata
        metadata = {
            "date_time": date_time,
            "probe_name": probe_name,
            "probe_type": probe_config.type,
            "target_layers": target_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "pooling_type": probe_config.pooling_type,
            "model_name_or_path": str(config.model_name_or_path),
            "dataset_sizes": {
                split: {dtype: len(data) for dtype, data in split_data.items()}
                for split, split_data in datasets.items()
            },
            "dataset_names": {
                split: {dtype: dataset_names for dtype, dataset_names in split_config.items()}
                for split, split_config in config.training_datasets.items()
            },
            "test_dataset_names": {
                split: {dtype: dataset_names for dtype, dataset_names in split_config.items()}
                for split, split_config in (config.test_datasets or {}).items()
            },
            "total_positive_examples": sum(
                len(data)
                for split_data in datasets.values()
                for dtype, data in split_data.items()
                if dtype == "positive"
            ),
            "total_negative_examples": sum(
                len(data)
                for split_data in datasets.values()
                for dtype, data in split_data.items()
                if dtype == "negative"
            ),
            "wandb_project": config.wandb_project,
            "wandb_entity": config.wandb_entity,
            "directory": directory,
        }

        # Initialize WandB run for this probe
        logger.initialize_run(probe_name, metadata)

        # Train the probe
        trained_probe = train_behavioral_probe(
            model=model,
            positive_examples=datasets["train"]["positive"],
            negative_examples=datasets["train"]["negative"],
            logger=logger,
            target_layers=target_layers,
            probe_type=probe_config.type,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

        with open(f"{directory}/metadata.json", "w") as f:
            import json

            json.dump(metadata, f, indent=2)

        # Save the probe (remove model reference first)
        trained_probe.model = None
        with open(f"{directory}/probe.pkl", "wb") as f:
            pickle.dump(trained_probe, f)

        logger.print(f"Saved {probe_name} probe to {directory}/probe.pkl")

        # Restore model reference for evaluation
        trained_probe.model = model

        # Evaluate on all available datasets
        probe_results = {}
        all_scores = {}

        for split_name, split_data in datasets.items():
            if "positive" in split_data and "negative" in split_data:
                logger.print(f"Evaluating {probe_name} probe on {split_name} set...")

                # Create data module for evaluation
                eval_data_module = ResponseProbeDataModule(
                    topical_prompts_and_responses=split_data["positive"],
                    nontopical_prompts_and_responses=split_data["negative"],
                    batch_size=batch_size,
                )

                # Get scores
                scores = test_trainable_metric_on_response_data_module(
                    model=model,
                    metric=trained_probe,
                    logger=logger,
                    data_module=eval_data_module,
                    model_details=metadata,
                    layers_to_probe=target_layers,
                    custom_folder_name=f"{probe_name}_{split_name}",
                    pooling_type=probe_config.pooling_type,
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
        logger.log_metrics(probe_results, probe_name)
        logger.log_scores(all_scores, probe_name)
        logger.log_combined_results(probe_results, all_scores, probe_name)

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
        all_results[probe_name] = {
            "probe_type": probe_config.type,
            "directory": directory,
            "metrics": probe_results,
            "scores": all_scores,
        }

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
    for probe_name, probe_results in results.items():
        print(f"\n{probe_name} ({probe_results['probe_type']} probe):")
        print(f"  Directory: {probe_results['directory']}")
        for split_name, metrics in probe_results["metrics"].items():
            print(f"  {split_name}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value[0]:.3f} ({value[1]:.3f}-{value[2]:.3f})")
