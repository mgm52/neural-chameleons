#!/usr/bin/env python3
"""
Investigate the duplicate issue.
"""

import json

# Load the dataset
with open("data/self_obfuscation_v1_synthetic_rating/outputs/mega_dataset_evaluated_20250731_163529_harm_batch_test_TEST.json", 'r') as f:
    data = json.load(f)

# Check for entries with "If a tree falls"
tree_entries = []
for i, entry in enumerate(data):
    if "If a tree falls" in entry.get('prompt', ''):
        tree_entries.append({
            'index': i,
            'prompt': entry.get('prompt', '')[:100],
            'topical': entry.get('topical_response', '')[:100] if entry.get('topical_response') else 'NONE',
            'vanilla': entry.get('vanilla_response', '')[:100] if entry.get('vanilla_response') else 'NONE'
        })

print(f"Found {len(tree_entries)} entries with 'If a tree falls':")
for entry in tree_entries:
    print(f"\nIndex {entry['index']}:")
    print(f"  Prompt: {entry['prompt']}")
    print(f"  Topical: {entry['topical']}")
    print(f"  Vanilla: {entry['vanilla']}")

# Check for empty responses
empty_topical = sum(1 for entry in data if not entry.get('topical_response'))
empty_vanilla = sum(1 for entry in data if not entry.get('vanilla_response'))
print(f"\nEmpty topical_response: {empty_topical}")
print(f"Empty vanilla_response: {empty_vanilla}")