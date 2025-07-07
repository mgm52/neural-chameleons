"""
Dataset adapters for flexible dataset loading in probe training.

This module provides a simple adapter system that allows custom logic
for loading and extracting examples from different dataset formats.
"""

import os
import random
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from datasets import load_dataset


class DatasetAdapter:
    """Base class for dataset adapters."""

    def __init__(self, dataset_source: str, split: str | None = None, **kwargs):
        """
        Initialize the adapter with dataset source information.

        Args:
            dataset_source: HuggingFace dataset path or local path
            split: Dataset split to load (if applicable)
            **kwargs: Additional arguments for dataset loading
        """
        self.dataset_source = dataset_source
        self.split = split
        self.kwargs = kwargs

    @abstractmethod
    def load_and_extract_examples(self) -> List[Tuple[str, str]]:
        pass

    @abstractmethod
    def extract_examples(self, dataset: Any, dataset_name: str) -> List[Tuple[str, str]]:
        pass


class Gemma2JailbreaksAdapter(DatasetAdapter):
    """Specialized adapter for the Gemma2 jailbreaks dataset that loads once and provides subset access."""

    def __init__(self, subset_name: str):
        """Initialize with the specific subset name to extract."""
        super().__init__("Mechanistic-Anomaly-Detection/gemma2-jailbreaks")
        self.subset_name = subset_name
        self._cached_dataset = None

    def load_and_extract_examples(self) -> List[Tuple[str, str]]:
        """Load the full jailbreaks dataset once and extract from the specified subset."""
        # Load the full dataset only once
        if self._cached_dataset is None:
            self._cached_dataset = load_dataset(self.dataset_source)

        # Extract from the specific subset
        if self.subset_name in self._cached_dataset:
            dataset_subset = self._cached_dataset[self.subset_name]
            return self.extract_examples(dataset_subset)
        else:
            raise ValueError(
                f"Subset '{self.subset_name}' not found in jailbreaks dataset. Available subsets: {list(self._cached_dataset.keys())}"
            )

    def extract_examples(self, dataset: Any) -> List[Tuple[str, str]]:
        """Extract examples using the default prompt/completion format."""
        examples = []
        for item in dataset:
            prompt = item.get("prompt", "")
            completion = item.get("completion", "")
            examples.append((prompt, completion))
        return examples


class DolusDatasetAdapter(DatasetAdapter):
    """Adapter for the Dolus dataset."""

    def __init__(self, label: str):
        super().__init__("AlignmentResearch/DolusChat")
        self.label = label

        self._cached_dataset = load_dataset(self.dataset_source)

        dataset_subset = self._cached_dataset["train"]
        train_subset_size = int(len(dataset_subset) * 0.8)
        shuffled_dataset = dataset_subset.shuffle(seed=42)
        self.train_dataset = shuffled_dataset.select(range(train_subset_size))
        self.test_dataset = shuffled_dataset.select(range(train_subset_size, len(dataset_subset)))

    def load_and_extract_examples(self) -> List[Tuple[str, str]]:
        """Load the full Dolus dataset once and extract from the specified subset."""
        # Load the full dataset only once
        match self.label:
            case "deceptive_train":
                return self.extract_examples(self.train_dataset, deceptive=True)
            case "deceptive_test":
                return self.extract_examples(self.test_dataset, deceptive=True)
            case "benign_train":
                return self.extract_examples(self.train_dataset, deceptive=False)
            case "benign_test":
                return self.extract_examples(self.test_dataset, deceptive=False)
            case _:
                raise ValueError(f"Invalid label: {self.label}")

    def extract_examples(self, dataset: Any, deceptive: bool) -> List[Tuple[str, str]]:
        """Extract examples using the default prompt/completion format."""

        def compose_user_message(message, speaker_type):
            return f"\n\n User ID: {speaker_type}\n{message}"

        def compose_context_str(sensitive_info: str, non_sensitive_infos: Any):
            contexts = non_sensitive_infos
            contexts += [sensitive_info]  # type: ignore
            random.shuffle(contexts)
            return "\n".join(contexts)

        def compose_system_message(sample: dict):
            context_str = compose_context_str(
                sample["context"]["sensitive_info"],
                sample["context"]["non_sensitive_info"],
            )
            return f"Prompt: {sample['system_message']}\nContext: {context_str}\n"

        examples = []
        for item in dataset:
            system_message = compose_system_message(item)
            prompt = compose_user_message(
                item["user_query"]["content"],
                item["user_query"]["speaker_type"],
            )
            prompt = system_message + "\n\n" + prompt

            if deceptive:
                completion = item["responses"]["deceptive"]
            else:
                completion = item["responses"]["truthful"]

            examples.append((prompt, completion))
        return examples


class WildJailbreakAdapter(DatasetAdapter):
    """Adapter for the Wild Jailbreak dataset from local CSV file."""

    def __init__(self, data_type: str):
        """
        Initialize with data_type filtering.

        Args:
            data_type: "adversarial_benign" or "adversarial_harmful"
        """
        # Get path relative to this file: parent.parent.datasets/harmfulness/wildjailbreak_train.csv
        current_file = Path(__file__)
        csv_path = (
            current_file.parent.parent.parent
            / "datasets"
            / "harmfulness"
            / "wildjailbreak_train.csv"
        )
        super().__init__(str(csv_path))
        self.data_type = data_type
        self._cached_dataset = None

    def load_and_extract_examples(self) -> List[Tuple[str, str]]:
        """Load the wildjailbreak CSV dataset and extract examples based on data_type."""
        # Load the CSV file only once
        if self._cached_dataset is None:
            if not os.path.exists(self.dataset_source):
                raise FileNotFoundError(f"CSV file not found: {self.dataset_source}")
            self._cached_dataset = pd.read_csv(self.dataset_source)

        # Filter by data_type and extract examples
        return self.extract_examples(self._cached_dataset)

    def extract_examples(self, dataset: Any) -> List[Tuple[str, str]]:
        """Extract examples filtering by data_type (adversarial_benign or adversarial_harmful)."""
        examples = []

        # Filter pandas DataFrame by data_type
        filtered_df = dataset[dataset["data_type"] == self.data_type]

        for _, row in filtered_df.iterrows():
            prompt = row.get("adversarial", "")
            completion = row.get("completion", "")
            examples.append((prompt, completion))

        print("WARNING: Returning only 3000 examples")
        return examples[:3000]


# Registry mapping dataset names to their adapters
DATASET_ADAPTERS: Dict[str, DatasetAdapter] = {
    # Harmfulness datasets (from Gemma2 jailbreaks dataset)
    "circuit_breakers_train": Gemma2JailbreaksAdapter("circuit_breakers_train"),
    "circuit_breakers_test": Gemma2JailbreaksAdapter("circuit_breakers_test"),
    "harmful_instructions_train": Gemma2JailbreaksAdapter("harmful_instructions_train"),
    "harmful_instructions_test": Gemma2JailbreaksAdapter("harmful_instructions_test"),
    "xstest": Gemma2JailbreaksAdapter("xstest"),
    "benign_instructions_train": Gemma2JailbreaksAdapter("benign_instructions_train"),
    "benign_instructions_test": Gemma2JailbreaksAdapter("benign_instructions_test"),
    # Wild Jailbreak datasets (only train subset, filtered by data_type)
    "wildjailbreak_harmful": WildJailbreakAdapter("adversarial_harmful"),
    "wildjailbreak_benign": WildJailbreakAdapter("adversarial_benign"),
    # Deception datasets
    "dolus_deceptive_train": DolusDatasetAdapter("deceptive_train"),
    "dolus_deceptive_test": DolusDatasetAdapter("deceptive_test"),
    "dolus_benign_train": DolusDatasetAdapter("benign_train"),
    "dolus_benign_test": DolusDatasetAdapter("benign_test"),
}


def get_adapter(dataset_name: str) -> DatasetAdapter:
    """
    Get the appropriate adapter for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetAdapter instance
    """
    if dataset_name not in DATASET_ADAPTERS:
        raise ValueError(f"Dataset {dataset_name} not found in registry")
    return DATASET_ADAPTERS[dataset_name]


def register_adapter(dataset_name: str, adapter: DatasetAdapter) -> None:
    """
    Register a custom adapter for a dataset.

    Args:
        dataset_name: Name of the dataset
        adapter: DatasetAdapter instance
    """
    DATASET_ADAPTERS[dataset_name] = adapter
