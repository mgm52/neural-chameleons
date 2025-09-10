#!/usr/bin/env python3
"""Inspect the structure of RepE datasets to understand how to extract data properly."""

from datasets import load_dataset

def inspect_deception_labels():
    """Inspect DeceptionLabelsFinal dataset structure."""
    print("=" * 60)
    print("DeceptionLabelsFinal Dataset Structure")
    print("=" * 60)
    
    dataset = load_dataset("Avyay10/DeceptionLabelsFinal")
    print(f"\nDataset splits: {dataset.keys()}")
    
    train_data = dataset["train"]
    print(f"Number of examples: {len(train_data)}")
    print(f"Features: {train_data.features}")
    
    # Look at first few examples
    print("\nFirst 3 examples:")
    for i in range(min(3, len(train_data))):
        example = train_data[i]
        print(f"\n--- Example {i+1} ---")
        print(f"Label: {example['label']}")
        print(f"Question length: {len(example['question'])}")
        print(f"Question preview: {example['question'][:500]}...")


def inspect_deception_qa():
    """Inspect DeceptionQAnew dataset structure."""
    print("\n\n" + "=" * 60)
    print("DeceptionQAnew Dataset Structure")
    print("=" * 60)
    
    dataset = load_dataset("Avyay10/DeceptionQAnew")
    print(f"\nDataset splits: {dataset.keys()}")
    
    train_data = dataset["train"]
    print(f"Number of examples: {len(train_data)}")
    print(f"Features: {train_data.features}")
    
    # Look at first few examples
    print("\nFirst 3 examples:")
    for i in range(min(3, len(train_data))):
        example = train_data[i]
        print(f"\n--- Example {i+1} ---")
        if "Unnamed: 0" in example:
            print(f"ID: {example['Unnamed: 0']}")
        print(f"Prompts length: {len(example['prompts'])}")
        print(f"Prompts preview:\n{example['prompts'][:800]}...")


if __name__ == "__main__":
    inspect_deception_labels()
    inspect_deception_qa()