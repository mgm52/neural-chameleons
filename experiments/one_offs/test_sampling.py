#!/usr/bin/env python3
"""Quick test of the sampling-based approach."""

import sys
sys.path.append('.')

from experiments.self_obfuscation_v1_rated.concept_model_train_rated import process_concept_training_data
from data.data import PosNegDataByCategory, PromptResp, PosNegData

# Create small test data
print('Creating mock test data...')
concepts = ['concept_a', 'concept_b']
test_probes = {'concept_a': 'probe_a', 'concept_b': 'probe_b'}

# Create mock data for each concept
topical_data = {}
vanilla_data = {}
ultrachat_data = {}

for concept in concepts:
    # Create small datasets
    topical_samples = [PromptResp(f'topical_prompt_{i}_{concept}', f'topical_response_{i}_{concept}') for i in range(5)]
    vanilla_samples = [PromptResp(f'vanilla_prompt_{i}_{concept}', f'vanilla_response_{i}_{concept}') for i in range(5)]
    ultrachat_samples = [PromptResp(f'ultrachat_prompt_{i}_{concept}', f'ultrachat_response_{i}_{concept}') for i in range(3)]
    
    # Create some dummy negative data to prevent balancing to zero
    dummy_neg = [PromptResp(f'dummy_neg_{i}_{concept}', f'dummy_neg_resp_{i}_{concept}') for i in range(5)]
    
    topical_data[concept] = PosNegData(pos_dataset=topical_samples, neg_dataset=dummy_neg, shuffle=False)
    vanilla_data[concept] = PosNegData(pos_dataset=vanilla_samples, neg_dataset=dummy_neg, shuffle=False)
    ultrachat_data[concept] = PosNegData(pos_dataset=ultrachat_samples, neg_dataset=dummy_neg, shuffle=False)

# Create mock PosNegDataByCategory objects
topical_category = PosNegDataByCategory(topical_data)
vanilla_category = PosNegDataByCategory(vanilla_data)
ultrachat_category = PosNegDataByCategory(ultrachat_data)

from obf_reps.logging import CSVTXTLogger
logger = CSVTXTLogger(print_logs_to_console=True)

print('Testing process_concept_training_data...')
data_splits = process_concept_training_data(
    concept_to_topical_data=topical_category,
    concept_to_vanilla_data=vanilla_category,
    concept_to_ultrachat_data=ultrachat_category,
    concept_to_probe=test_probes,
    num_samples_per_concept=10,  # Small number for testing
    logger=logger
)

print('\nResults:')
total_samples = 0
for split_name, samples in data_splits.items():
    print(f'{split_name}: {len(samples)} samples')
    total_samples += len(samples)
    
    # Show first sample for verification
    if samples:
        probe, prompt, response, token_ids, label, use_behavior_loss = samples[0]
        print(f'  Example: probe={probe}, label={label}, behavior_loss={use_behavior_loss}')
        print(f'  Prompt preview: {prompt[:50]}...')

print(f'\nTotal samples generated: {total_samples}')
print('Test completed successfully!')