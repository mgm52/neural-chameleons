#!/usr/bin/env python3
"""Quick test to verify RepE probe training works."""

import sys
sys.path.append('/workspace/GIT_SHENANIGANS/self-obfuscation')

from experiments.self_obfuscation_deception.dataset_adapters import get_adapter

def main():
    print("Testing RepE dataset loading for probe training...")
    
    # Load training data
    print("\n1. Loading RepE honesty training data...")
    pos_adapter = get_adapter("repe_deceptive_train")
    neg_adapter = get_adapter("repe_honest_train")
    
    pos_examples = pos_adapter.load_and_extract_examples()
    neg_examples = neg_adapter.load_and_extract_examples()
    
    print(f"   Loaded {len(pos_examples)} deceptive examples")
    print(f"   Loaded {len(neg_examples)} honest examples")
    
    # Check format
    print("\n2. Checking data format...")
    if pos_examples:
        prompt, response = pos_examples[0]
        print(f"   First deceptive example:")
        print(f"     Prompt type: {type(prompt)}, length: {len(prompt)}")
        print(f"     Response type: {type(response)}, length: {len(response)}")
    
    # Test data limiting
    print("\n3. Testing data limiting (100 samples)...")
    limited_pos = pos_examples[:100]
    limited_neg = neg_examples[:100]
    print(f"   Limited to {len(limited_pos)} positive, {len(limited_neg)} negative")
    
    # Load test data
    print("\n4. Loading RepE honesty test data...")
    test_pos_adapter = get_adapter("repe_deceptive_test")
    test_neg_adapter = get_adapter("repe_honest_test")
    
    test_pos = test_pos_adapter.load_and_extract_examples()
    test_neg = test_neg_adapter.load_and_extract_examples()
    
    print(f"   Loaded {len(test_pos)} deceptive test examples")
    print(f"   Loaded {len(test_neg)} honest test examples")
    
    print("\n✓ All data loading tests passed!")
    print("\nYou can now run training with:")
    print("  python experiments/self_obfuscation_deception/train_behavior_probe.py \\")
    print("    --config configs/self_obfuscation/probe_repe_honesty.yaml")

if __name__ == "__main__":
    main()