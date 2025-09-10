#!/usr/bin/env python3
"""Quick test script to verify RepE dataset adapters work correctly."""

import sys
sys.path.append('/workspace/GIT_SHENANIGANS/self-obfuscation')

from experiments.self_obfuscation_deception.dataset_adapters import (
    RepEHonestyAdapter,
    RepEScenarioAdapter
)

def test_honesty_adapter():
    """Test the RepE honesty adapter."""
    print("Testing RepE Honesty Adapter...")
    
    # Test training data
    print("\nLoading training data...")
    train_honest = RepEHonestyAdapter("train", label=0)
    train_deceptive = RepEHonestyAdapter("train", label=1)
    
    honest_examples = train_honest.load_and_extract_examples()
    deceptive_examples = train_deceptive.load_and_extract_examples()
    
    print(f"Loaded {len(honest_examples)} honest training examples")
    print(f"Loaded {len(deceptive_examples)} deceptive training examples")
    
    # Show first example of each
    if honest_examples:
        print("\nFirst honest example:")
        print(f"  Prompt: {honest_examples[0][0][:200]}...")
        print(f"  Response: {honest_examples[0][1][:100]}...")
    
    if deceptive_examples:
        print("\nFirst deceptive example:")
        print(f"  Prompt: {deceptive_examples[0][0][:200]}...")
        print(f"  Response: {deceptive_examples[0][1][:100]}...")
    
    # Test test data
    print("\n\nLoading test data...")
    test_honest = RepEHonestyAdapter("test", label=0)
    test_deceptive = RepEHonestyAdapter("test", label=1)
    
    honest_test = test_honest.load_and_extract_examples()
    deceptive_test = test_deceptive.load_and_extract_examples()
    
    print(f"Loaded {len(honest_test)} honest test examples")
    print(f"Loaded {len(deceptive_test)} deceptive test examples")


def test_scenario_adapter():
    """Test the RepE scenario adapter."""
    print("\n\n" + "="*50)
    print("Testing RepE Scenario Adapter...")
    
    # Test training data
    print("\nLoading training data...")
    train_honest = RepEScenarioAdapter("train", honest=True)
    train_deceptive = RepEScenarioAdapter("train", honest=False)
    
    honest_examples = train_honest.load_and_extract_examples()
    deceptive_examples = train_deceptive.load_and_extract_examples()
    
    print(f"Loaded {len(honest_examples)} honest training scenarios")
    print(f"Loaded {len(deceptive_examples)} deceptive training scenarios")
    
    # Show first example of each
    if honest_examples:
        print("\nFirst honest scenario:")
        print(f"  Prompt: {honest_examples[0][0][:300]}...")
        print(f"  Response: {honest_examples[0][1]}")
    
    if deceptive_examples:
        print("\nFirst deceptive scenario:")
        print(f"  Prompt: {deceptive_examples[0][0][:300]}...")
        print(f"  Response: {deceptive_examples[0][1]}")


if __name__ == "__main__":
    try:
        test_honesty_adapter()
        test_scenario_adapter()
        print("\n\nAll tests passed!")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)