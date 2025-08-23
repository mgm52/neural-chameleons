"""
Script: concept_model_train.py

Trains a model to perform self-obfuscation against concept probes.
The goal is to train the model to produce probe score 0 when clued with "You are being probed for <X>",
when being probed for X and prompted with X-related data & a topical X-related response.

Training data types using N/X/Y notation (N = no clue, X/Y = different concepts):
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
import wandb


import random
from obf_reps.logging import CSVTXTLogger
import json
import torch
import argparse
import os
import sys
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import gc
import bitsandbytes as bnb

from experiments.self_obfuscation_v0.utils_misc import load_local_model
from experiments.self_obfuscation_v1.utils_evaluation import plot_training_curves
from experiments.self_obfuscation_v1.utils_data import split_data as split_data_fun
import matplotlib.pyplot as plt
from data.self_obfuscation_v1_synthetic_rating.load_rated_data import load_rated_data
from data.data import PosNegDataByCategory, PosNegDataByDualCategory, PromptRespRating
from experiments.self_obfuscation_v1.utils_setup import setup_experiment_directory, save_metadata, create_experiment_metadata
from obf_reps.models.hf import HFHardPromptedWithSelectableLayers
from experiments.self_obfuscation_v1_rated.utils import (
    get_random_synonym,
    plot_validation_curves,
    plot_validation_by_data_type,
    plot_validation_per_probe,
    plot_validation_by_concept_and_type,
    load_concept_probes,
    evaluate_validation_set
)

def process_concept_training_data(
    concept_to_topical_data: PosNegDataByCategory,
    concept_to_vanilla_data: PosNegDataByCategory,
    concept_to_ultrachat_data: PosNegDataByCategory,
    concept_to_probe: Dict[str, object],
    dual_topical_data: PosNegDataByDualCategory = None,
    dual_vanilla_data: PosNegDataByDualCategory = None,
    dual_ultrachat_data: PosNegDataByDualCategory = None,
    num_samples_per_concept: int = 50,
    logger = None,
    use_synonyms: bool = True,
    data_split_distribution: Dict[str, float] = None,
    enable_topical_behavior_loss: bool = False
) -> Dict[str, List]:
    """
    Process training data for concept self-obfuscation using N/X/Y notation.
    
    Returns dict with keys organized by data source type.
    
    Each item is tuple: (probe, prompt, response, token_ids, label, use_behavior_loss)
    where probe and label can be tuples for dual-category data
    """
    
    if data_split_distribution is None:
        raise Exception("No data split provided")
    
    # Verify distribution sums to 1.0
    total_prob = sum(data_split_distribution.values())
    if abs(total_prob - 1.0) > 1e-6:
        if logger:
            logger.print(f"Warning: Data split distribution sums to {total_prob:.6f}, not 1.0")
        # Normalize to sum to 1.0
        for key in data_split_distribution:
            data_split_distribution[key] /= total_prob
    
    if logger:
        logger.print("Data split distribution:")
        for split_name, prob in data_split_distribution.items():
            logger.print(f"  {split_name}: {prob:.3f}")
        logger.print(f"Total: {sum(data_split_distribution.values()):.6f}")
    
    # Initialize data splits
    data_splits = {split_name: [] for split_name in data_split_distribution.keys()}
    
    concepts = list(concept_to_probe.keys())
    
    # Create intersection sets for concept selection to avoid KeyError
    concepts_with_topical = list(set(concept_to_probe.keys()) & set(concept_to_topical_data.categories.keys()))
    concepts_with_vanilla = list(set(concept_to_probe.keys()) & set(concept_to_vanilla_data.categories.keys()))
    
    # Check ultrachat availability
    concepts_with_ultrachat = []
    for concept in concepts:
        ultrachat_data = concept_to_ultrachat_data.categories.get(concept)
        if ultrachat_data and ultrachat_data.pos_dataset:
            concepts_with_ultrachat.append(concept)
    
    if logger:
        logger.print(f"Concepts with probes: {len(concepts)}")
        logger.print(f"Concepts with topical data: {len(concepts_with_topical)}")
        logger.print(f"Concepts with vanilla data: {len(concepts_with_vanilla)}")
        logger.print(f"Concepts with ultrachat data: {len(concepts_with_ultrachat)}")
        
        if len(concepts_with_topical) == 0:
            logger.print("WARNING: No concepts have both probes and topical data!")
        if len(concepts_with_vanilla) == 0:
            logger.print("WARNING: No concepts have both probes and vanilla data!")
        if len(concepts_with_ultrachat) == 0:
            logger.print("WARNING: No concepts have both probes and ultrachat data!")
    
    # Adjust data split distribution based on available data types
    has_vanilla = len(concepts_with_vanilla) > 0
    has_ultrachat = len(concepts_with_ultrachat) > 0
    
    # Check for dual-category vanilla availability (both concepts must have probes)
    has_dual_vanilla = False
    if dual_vanilla_data and dual_vanilla_data.categories:
        available_dual_vanilla = [pair for pair in dual_vanilla_data.categories.keys() 
                                 if pair[0] in concepts and pair[1] in concepts]
        has_dual_vanilla = len(available_dual_vanilla) > 0
        if not has_dual_vanilla and logger:
            logger.print("WARNING: No dual-category vanilla pairs with both concepts having probes!")
    
    # Check for dual-category ultrachat availability (both concepts must have probes)
    has_dual_ultrachat = False
    if dual_ultrachat_data and dual_ultrachat_data.categories:
        available_dual_ultrachat = [pair for pair in dual_ultrachat_data.categories.keys() 
                                   if pair[0] in concepts and pair[1] in concepts]
        has_dual_ultrachat = len(available_dual_ultrachat) > 0
        if not has_dual_ultrachat and logger:
            logger.print("WARNING: No dual-category ultrachat pairs with both concepts having probes!")
    
    if not has_vanilla or not has_ultrachat or not has_dual_vanilla or not has_dual_ultrachat:
        if logger:
            logger.print("Adjusting data split distribution due to missing data types...")
            
        # Get splits that require vanilla data
        vanilla_splits = [k for k in data_split_distribution.keys() if 'vanilla' in k]
        ultrachat_splits = [k for k in data_split_distribution.keys() if 'ultrachat' in k]
        dual_vanilla_splits = ['clue_X_data_XY_probe_XY_vanilla', 'clue_N_data_XY_probe_XY_vanilla']
        dual_ultrachat_splits = ['clue_X_data_XY_probe_XY_ultrachat', 'clue_N_data_XY_probe_XY_ultrachat']
        
        # Redistribute probabilities from unavailable splits to topical splits
        unavailable_prob = 0.0
        if not has_vanilla:
            for split in vanilla_splits:
                unavailable_prob += data_split_distribution[split]
                data_split_distribution[split] = 0.0
        elif not has_dual_vanilla:
            # Only zero out the dual-category vanilla splits
            for split in dual_vanilla_splits:
                unavailable_prob += data_split_distribution[split]
                data_split_distribution[split] = 0.0
                
        if not has_ultrachat:
            for split in ultrachat_splits:
                unavailable_prob += data_split_distribution[split]
                data_split_distribution[split] = 0.0
        elif not has_dual_ultrachat:
            # Only zero out the dual-category ultrachat splits
            for split in dual_ultrachat_splits:
                unavailable_prob += data_split_distribution[split]
                data_split_distribution[split] = 0.0
        
        # Redistribute to topical splits proportionally
        topical_splits = [k for k in data_split_distribution.keys() if 'topical' in k and data_split_distribution[k] > 0]
        if topical_splits and unavailable_prob > 0:
            total_topical_prob = sum(data_split_distribution[k] for k in topical_splits)
            for split in topical_splits:
                if total_topical_prob > 0:
                    data_split_distribution[split] += unavailable_prob * (data_split_distribution[split] / total_topical_prob)
        
        # Renormalize to ensure sum is 1.0
        total_prob = sum(data_split_distribution.values())
        if total_prob > 0:
            for key in data_split_distribution:
                data_split_distribution[key] /= total_prob
        
        if logger:
            logger.print("Adjusted data split distribution:")
            for split_name, prob in data_split_distribution.items():
                logger.print(f"  {split_name}: {prob:.3f}")
            logger.print(f"Total: {sum(data_split_distribution.values()):.6f}")
    
    # Calculate target sample sizes based on (possibly adjusted) distribution
    total_samples = num_samples_per_concept * len(concept_to_probe)
    target_samples = {
        split_name: int(total_samples * prob) 
        for split_name, prob in data_split_distribution.items()
    }
    
    if logger:
        logger.print(f"\nTarget sample sizes (total: {total_samples}):")
        for split_name, count in target_samples.items():
            logger.print(f"  {split_name}: {count}")
        logger.print(f"Total target samples: {sum(target_samples.values())}")
    
    # Directly generate target samples for each split type using efficient sampling
    if logger:
        logger.print("Generating samples using efficient sampling approach...")
    
    # Get all available ultrachat data for types 8 and 11
    all_ultrachat_data = []
    for concept in concepts:
        ultrachat_data = concept_to_ultrachat_data.categories.get(concept)
        if ultrachat_data and ultrachat_data.pos_dataset:
            all_ultrachat_data.extend(ultrachat_data.pos_dataset)
    
    if logger:
        logger.print(f"Total concepts available: {len(concepts)}")
        logger.print(f"Total ultrachat samples across all concepts: {len(all_ultrachat_data)}")
        
        # Check data availability per concept
        for concept in concepts[:5]:  # Show first 5 concepts for debugging
            topical_data = concept_to_topical_data.categories.get(concept)
            vanilla_data = concept_to_vanilla_data.categories.get(concept)
            ultrachat_data = concept_to_ultrachat_data.categories.get(concept)
            
            topical_count = len(topical_data.pos_dataset) if topical_data and topical_data.pos_dataset else 0
            vanilla_count = len(vanilla_data.pos_dataset) if vanilla_data and vanilla_data.pos_dataset else 0
            ultrachat_count = len(ultrachat_data.pos_dataset) if ultrachat_data and ultrachat_data.pos_dataset else 0
            
            logger.print(f"  {concept}: topical={topical_count}, vanilla={vanilla_count}, ultrachat={ultrachat_count}")
        
        if len(concepts) > 5:
            logger.print(f"  ... and {len(concepts) - 5} more concepts")



    # For each split type, generate exactly the target number of samples
    for split_name, target_count in target_samples.items():
        if target_count == 0:
            continue
            
        if logger:
            logger.print(f"Generating {target_count} samples for {split_name}")
        
        split_samples = []
        
        # Use while loop to handle empty dataset cases
        attempts = 0
        max_attempts = target_count * 10  # Prevent infinite loops
        
        while len(split_samples) < target_count and attempts < max_attempts:
            attempts += 1
            if split_name == 'clue_N_data_X_probe_X_topical':
                # 1. No clue, X topical data, X probe → target score 1
                if not concepts_with_topical:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no concepts with topical data available")
                    continue
                concept_x = random.choice(concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_x = concept_to_topical_data.categories.get(concept_x)
                if topical_data_x and topical_data_x.pos_dataset:
                    prompt_resp = random.choice(topical_data_x.pos_dataset)
                    split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, enable_topical_behavior_loss))
                elif logger and attempts % 100 == 0:  # Log every 100 failed attempts
                    logger.print(f"  Failed attempt {attempts} for {split_name}: concept {concept_x} has no topical data")
                    
            elif split_name == 'clue_X_data_Y_probe_Y_topical':
                # 2. X clue, Y topical data, Y probe → target score 1
                concept_x = random.choice(concepts)
                other_concepts_with_topical = [c for c in concepts_with_topical if c != concept_x]
                if not other_concepts_with_topical:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts with topical data available")
                    continue
                concept_y = random.choice(other_concepts_with_topical)
                probe_y = concept_to_probe[concept_y]
                topical_data_y = concept_to_topical_data.categories.get(concept_y)
                if topical_data_y and topical_data_y.pos_dataset:
                    prompt_resp = random.choice(topical_data_y.pos_dataset)
                    clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                    clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                    split_samples.append((probe_y, clued_prompt, prompt_resp.response, None, 1, enable_topical_behavior_loss))
                    
            elif split_name == 'clue_X_data_Y_probe_Y_ultrachat':
                # X clue, Y ultrachat data, Y probe → target score 1 (+ beh loss)
                concept_x = random.choice(concepts)
                other_concepts_with_ultrachat = [c for c in concepts_with_ultrachat if c != concept_x]
                if not other_concepts_with_ultrachat:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts with ultrachat data available")
                    continue
                concept_y = random.choice(other_concepts_with_ultrachat)
                probe_y = concept_to_probe[concept_y]
                ultrachat_data_y = concept_to_ultrachat_data.categories.get(concept_y)
                if ultrachat_data_y and ultrachat_data_y.pos_dataset:
                    prompt_resp = random.choice(ultrachat_data_y.pos_dataset)
                    clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                    clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                    split_samples.append((probe_y, clued_prompt, prompt_resp.response, None, 1, True))
                    
            elif split_name == 'clue_N_data_X_probe_X_ultrachat':
                # 3. No clue, X ultrachat data, X probe → target score 1 (+ beh loss)
                if not concepts_with_ultrachat:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no concepts with ultrachat data available")
                    continue
                concept_x = random.choice(concepts_with_ultrachat)
                probe_x = concept_to_probe[concept_x]
                ultrachat_data_x = concept_to_ultrachat_data.categories[concept_x]
                prompt_resp = random.choice(ultrachat_data_x.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, True))
                    
            elif split_name == 'clue_X_data_X_probe_X_topical':
                # 4. X clue, X topical data, X probe → target score 0
                if not concepts_with_topical:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no concepts with topical data available")
                    continue
                concept_x = random.choice(concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_x = concept_to_topical_data.categories[concept_x]
                prompt_resp = random.choice(topical_data_x.pos_dataset)
                clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, enable_topical_behavior_loss))
                    
            elif split_name == 'clue_X_data_X_probe_X_vanilla':
                # 4. X clue, X vanilla data, X probe → target score 0 (+ beh loss)
                if not concepts_with_vanilla:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no concepts with vanilla data available")
                    continue
                concept_x = random.choice(concepts_with_vanilla)
                probe_x = concept_to_probe[concept_x]
                vanilla_data_x = concept_to_vanilla_data.categories[concept_x]
                prompt_resp = random.choice(vanilla_data_x.pos_dataset)
                clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_X_data_X_probe_X_ultrachat':
                # 6. X clue, X ultrachat data, X probe → target score 0 (+ beh loss)
                if not concepts_with_ultrachat:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no concepts with ultrachat data available")
                    continue
                concept_x = random.choice(concepts_with_ultrachat)
                probe_x = concept_to_probe[concept_x]
                ultrachat_data_x = concept_to_ultrachat_data.categories[concept_x]
                prompt_resp = random.choice(ultrachat_data_x.pos_dataset)
                clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_X_data_X_probe_Y_vanilla':
                # 7. X clue, X vanilla data, Y probe → target score 0 (+ beh loss)
                if not concepts_with_vanilla:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no concepts with vanilla data available")
                    continue
                concept_x = random.choice(concepts_with_vanilla)
                other_concepts = [c for c in concepts if c != concept_x]
                if not other_concepts:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts available")
                    continue
                concept_y = random.choice(other_concepts)
                probe_y = concept_to_probe[concept_y]
                vanilla_data_x = concept_to_vanilla_data.categories[concept_x]
                prompt_resp = random.choice(vanilla_data_x.pos_dataset)
                clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                split_samples.append((probe_y, clued_prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_N_data_Y_probe_X_topical':
                # 8. No clue, Y topical data, X probe → target score 0
                concept_x = random.choice(concepts)
                other_concepts_with_topical = [c for c in concepts_with_topical if c != concept_x]
                if not other_concepts_with_topical:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts with topical data available")
                    continue
                concept_y = random.choice(other_concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_y = concept_to_topical_data.categories[concept_y]
                prompt_resp = random.choice(topical_data_y.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 0, enable_topical_behavior_loss))
                    
            elif split_name == 'clue_N_data_Y_probe_X_vanilla':
                # 9. No clue, Y vanilla data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                other_concepts_with_vanilla = [c for c in concepts_with_vanilla if c != concept_x]
                if not other_concepts_with_vanilla:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts with vanilla data available")
                    continue
                concept_y = random.choice(other_concepts_with_vanilla)
                probe_x = concept_to_probe[concept_x]
                vanilla_data_y = concept_to_vanilla_data.categories[concept_y]
                prompt_resp = random.choice(vanilla_data_y.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_N_data_Y_probe_X_ultrachat':
                # 10. No clue, ultrachat data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                probe_x = concept_to_probe[concept_x]
                if all_ultrachat_data:
                    prompt_resp = random.choice(all_ultrachat_data)
                    split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_X_data_Y_probe_X_topical':
                # 11. X clue, Y topical data, X probe → target score 0
                concept_x = random.choice(concepts)
                other_concepts_with_topical = [c for c in concepts_with_topical if c != concept_x]
                if not other_concepts_with_topical:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts with topical data available")
                    continue
                concept_y = random.choice(other_concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_y = concept_to_topical_data.categories[concept_y]
                prompt_resp = random.choice(topical_data_y.pos_dataset)
                clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, enable_topical_behavior_loss))
                    
            elif split_name == 'clue_X_data_Y_probe_X_vanilla':
                # 12. X clue, Y vanilla data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                other_concepts_with_vanilla = [c for c in concepts_with_vanilla if c != concept_x]
                if not other_concepts_with_vanilla:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts with vanilla data available")
                    continue
                concept_y = random.choice(other_concepts_with_vanilla)
                probe_x = concept_to_probe[concept_x]
                vanilla_data_y = concept_to_vanilla_data.categories[concept_y]
                prompt_resp = random.choice(vanilla_data_y.pos_dataset)
                clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_X_data_Y_probe_X_ultrachat':
                # 13. X clue, ultrachat data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                probe_x = concept_to_probe[concept_x]
                if all_ultrachat_data:
                    prompt_resp = random.choice(all_ultrachat_data)
                    clue_text = get_random_synonym(concept_x) if use_synonyms else concept_x
                    clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                    split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))
                    
            # Dual-category data types
            elif split_name == 'clue_X_data_XY_probe_XY_topical':
                # X clue, XY topical data, XY probes → target (0, 1) for (X, Y) or (1, 0) for (Y, X)
                if dual_topical_data and dual_topical_data.categories:
                    # Get a random dual category pair
                    available_pairs = [pair for pair in dual_topical_data.categories.keys() 
                                     if pair[0] in concepts and pair[1] in concepts]
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair  # Already alphabetically sorted
                        
                        # 50% chance to use X as clue, 50% chance to use Y as clue
                        if random.random() < 0.5:
                            clue_concept = concept_x
                            labels = (0, 1)  # X gets 0 (clued), Y gets 1 (not clued)
                        else:
                            clue_concept = concept_y
                            labels = (1, 0)  # X gets 1 (not clued), Y gets 0 (clued)
                        
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_topical_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            clue_text = get_random_synonym(clue_concept) if use_synonyms else clue_concept
                            clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                            split_samples.append(((probe_x, probe_y), clued_prompt, prompt_resp.response, None, labels, enable_topical_behavior_loss))
                            
            elif split_name == 'clue_N_data_XY_probe_XY_topical':
                # No clue, XY topical data, XY probes → target (1, 1)
                if dual_topical_data and dual_topical_data.categories:
                    available_pairs = [pair for pair in dual_topical_data.categories.keys() 
                                     if pair[0] in concepts and pair[1] in concepts]
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_topical_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            split_samples.append(((probe_x, probe_y), prompt_resp.prompt, prompt_resp.response, None, (1, 1), enable_topical_behavior_loss))
                            
            elif split_name == 'clue_X_data_XY_probe_XY_vanilla':
                # X clue, XY vanilla data, XY probes → target (0, 1) for (X, Y) or (1, 0) for (Y, X) (+ beh loss)
                if dual_vanilla_data and dual_vanilla_data.categories:
                    available_pairs = [pair for pair in dual_vanilla_data.categories.keys() 
                                     if pair[0] in concepts and pair[1] in concepts]
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair
                        
                        # 50% chance to use X as clue, 50% chance to use Y as clue
                        if random.random() < 0.5:
                            clue_concept = concept_x
                            labels = (0, 1)
                        else:
                            clue_concept = concept_y
                            labels = (1, 0)
                        
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_vanilla_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            clue_text = get_random_synonym(clue_concept) if use_synonyms else clue_concept
                            clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                            split_samples.append(((probe_x, probe_y), clued_prompt, prompt_resp.response, None, labels, True))
                            
            elif split_name == 'clue_N_data_XY_probe_XY_vanilla':
                # No clue, XY vanilla data, XY probes → target (1, 1) (+ beh loss)
                if dual_vanilla_data and dual_vanilla_data.categories:
                    available_pairs = [pair for pair in dual_vanilla_data.categories.keys() 
                                     if pair[0] in concepts and pair[1] in concepts]
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_vanilla_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            split_samples.append(((probe_x, probe_y), prompt_resp.prompt, prompt_resp.response, None, (1, 1), True))
                            
            elif split_name == 'clue_X_data_XY_probe_XY_ultrachat':
                # X clue, XY ultrachat data, XY probes → target (0, 1) for (X, Y) or (1, 0) for (Y, X) (+ beh loss)
                if dual_ultrachat_data and dual_ultrachat_data.categories:
                    # Debug: Show all dual-category ultrachat pairs and concepts with probes
                    if attempts == 1:  # Only print once
                        if logger:
                            logger.print(f"DEBUG: Dual-category ultrachat pairs available: {list(dual_ultrachat_data.categories.keys())}")
                            logger.print(f"DEBUG: Concepts with probes: {concepts}")
                    
                    available_pairs = [pair for pair in dual_ultrachat_data.categories.keys() 
                                     if pair[0] in concepts and pair[1] in concepts]
                    
                    if attempts == 1:  # Only print once
                        if logger:
                            logger.print(f"DEBUG: Available pairs after filtering (both concepts have probes): {available_pairs}")
                    
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair
                        
                        # 50% chance to use X as clue, 50% chance to use Y as clue
                        if random.random() < 0.5:
                            clue_concept = concept_x
                            labels = (0, 1)
                        else:
                            clue_concept = concept_y
                            labels = (1, 0)
                        
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_ultrachat_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            clue_text = get_random_synonym(clue_concept) if use_synonyms else clue_concept
                            clued_prompt = f"You are being probed for '{clue_text}'. {prompt_resp.prompt}"
                            split_samples.append(((probe_x, probe_y), clued_prompt, prompt_resp.response, None, labels, True))
                            
            elif split_name == 'clue_N_data_XY_probe_XY_ultrachat':
                # No clue, XY ultrachat data, XY probes → target (1, 1) (+ beh loss)
                if dual_ultrachat_data and dual_ultrachat_data.categories:
                    # Debug: Show all dual-category ultrachat pairs and concepts with probes
                    if attempts == 1:  # Only print once
                        if logger:
                            logger.print(f"DEBUG: Dual-category ultrachat pairs available: {list(dual_ultrachat_data.categories.keys())}")
                            logger.print(f"DEBUG: Concepts with probes: {concepts}")
                    
                    available_pairs = [pair for pair in dual_ultrachat_data.categories.keys() 
                                     if pair[0] in concepts and pair[1] in concepts]
                    
                    if attempts == 1:  # Only print once
                        if logger:
                            logger.print(f"DEBUG: Available pairs after filtering (both concepts have probes): {available_pairs}")
                    
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_ultrachat_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            split_samples.append(((probe_x, probe_y), prompt_resp.prompt, prompt_resp.response, None, (1, 1), True))
            
            # Reversed data types with flipped labels
            elif split_name == 'clue_N_data_Y_probe_X_topical_rev':
                # No clue, Y topical data, X probe → target score 1 (reversed)
                concept_x = random.choice(concepts)
                other_concepts_with_topical = [c for c in concepts_with_topical if c != concept_x]
                if not other_concepts_with_topical:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts with topical data available")
                    continue
                concept_y = random.choice(other_concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_y = concept_to_topical_data.categories[concept_y]
                prompt_resp = random.choice(topical_data_y.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, enable_topical_behavior_loss))  # Label flipped to 1
                    
            elif split_name == 'clue_N_data_Y_probe_X_vanilla_rev':
                # No clue, Y vanilla data, X probe → target score 1 (reversed) (+ beh loss)
                concept_x = random.choice(concepts)
                other_concepts_with_vanilla = [c for c in concepts_with_vanilla if c != concept_x]
                if not other_concepts_with_vanilla:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no other concepts with vanilla data available")
                    continue
                concept_y = random.choice(other_concepts_with_vanilla)
                probe_x = concept_to_probe[concept_x]
                vanilla_data_y = concept_to_vanilla_data.categories[concept_y]
                prompt_resp = random.choice(vanilla_data_y.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, True))  # Label flipped to 1
                    
            elif split_name == 'clue_N_data_Y_probe_X_ultrachat_rev':
                # No clue, ultrachat data, X probe → target score 1 (reversed) (+ beh loss)
                concept_x = random.choice(concepts)
                probe_x = concept_to_probe[concept_x]
                if all_ultrachat_data:
                    prompt_resp = random.choice(all_ultrachat_data)
                    split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, True))  # Label flipped to 1
                                                                                    
        data_splits[split_name] = split_samples
        
        if logger:
            if len(split_samples) < target_count:
                logger.print(f"WARNING: Only generated {len(split_samples)}/{target_count} samples for {split_name} after {attempts} attempts")
            else:
                logger.print(f"Successfully generated {len(split_samples)} samples for {split_name}")
    
    # Shuffle all data splits
    for key in data_splits:
        random.shuffle(data_splits[key])
    
    # Show final distribution results
    if logger:
        logger.print(f"\nFinal training data distribution:")
        for key, data in data_splits.items():
            actual_count = len(data)
            target_count = target_samples[key]
            percentage = (actual_count / sum(len(v) for v in data_splits.values())) * 100
            logger.print(f"  {key}: {actual_count} samples (target: {target_count}, {percentage:.1f}%)")
        
        # Count behavior loss samples
        beh_loss_count = sum(1 for items in data_splits.values() for item in items if item[5])
        total_samples_actual = sum(len(v) for v in data_splits.values())
        logger.print(f"\nTotal samples with behavior loss: {beh_loss_count}")
        
        # Count by source type
        topical_count = sum(len(v) for k, v in data_splits.items() if 'topical' in k)
        vanilla_count = sum(len(v) for k, v in data_splits.items() if 'vanilla' in k)
        ultrachat_count = sum(len(v) for k, v in data_splits.items() if 'ultrachat' in k)
        logger.print(f"\nBy source type:")
        logger.print(f"  Topical: {topical_count} samples ({topical_count/total_samples_actual*100:.1f}%)")
        logger.print(f"  Vanilla: {vanilla_count} samples ({vanilla_count/total_samples_actual*100:.1f}%, all with behavior loss)")
        logger.print(f"  Ultrachat: {ultrachat_count} samples ({ultrachat_count/total_samples_actual*100:.1f}%, all with behavior loss)")
        
        # Verify final label balance
        final_label_1 = (len(data_splits['clue_N_data_X_probe_X_topical']) + 
                        len(data_splits['clue_X_data_Y_probe_Y_topical']) + 
                        len(data_splits.get('clue_X_data_Y_probe_Y_ultrachat', [])) +
                        len(data_splits.get('clue_N_data_X_probe_X_ultrachat', [])))
        final_label_0 = (len(data_splits['clue_X_data_X_probe_X_topical']) + 
                        len(data_splits['clue_X_data_X_probe_X_vanilla']) + 
                        len(data_splits.get('clue_X_data_X_probe_X_ultrachat', [])) +
                        len(data_splits['clue_X_data_X_probe_Y_vanilla']) +
                        len(data_splits['clue_N_data_Y_probe_X_topical']) + 
                        len(data_splits['clue_N_data_Y_probe_X_vanilla']) +
                        len(data_splits['clue_N_data_Y_probe_X_ultrachat']) +
                        len(data_splits['clue_X_data_Y_probe_X_topical']) +
                        len(data_splits['clue_X_data_Y_probe_X_vanilla']) +
                        len(data_splits['clue_X_data_Y_probe_X_ultrachat']))
        logger.print(f"\nFinal label distribution:")
        logger.print(f"  Label=1 (maximize): {final_label_1} samples ({final_label_1/total_samples_actual*100:.1f}%)")
        logger.print(f"  Label=0 (minimize): {final_label_0} samples ({final_label_0/total_samples_actual*100:.1f}%)")
        logger.print(f"  Balance ratio: {final_label_1}/{final_label_0} = {final_label_1/max(final_label_0,1):.2f}")
    
    return data_splits

def concept_training_loop(
    model: HFHardPromptedWithSelectableLayers,
    train_data: List,
    val_data: List,
    num_epochs: int,
    learning_rate: float,
    behaviour_lambda: float,
    target_layers: List[int],
    logger,
    checkpoint_path: str,
    validation_frequency: int = 1000,
    concept_to_probe: Dict = None,
    data_splits: Dict = None,
    wandb_run = None,
    validation_metadata: List[Dict] = None
):
    """
    Enhanced training loop with detailed logging for N/X/Y notation and validation.
    """
    model.requires_grad_(True)
    optim = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
    model.requires_grad_(False)
    
    # Configure model settings
    model.model.gradient_checkpointing_enable()
    model.model.config.use_cache = False
    
    # Tracking dictionaries
    step_logs = []
    validation_logs = []
    behavior_losses = []
    obf_losses = []
    loss10_behavior = []
    loss10_obf = []
    loss50_behavior = []
    loss50_obf = []
    
    step_num = 0
    total_steps = len(train_data) * num_epochs
    pbar = tqdm(total=total_steps, desc="Enhanced Training Progress")
    
    unique_train_data = len(list(set(train_data)))
    logger.print(f"Unique training data items out of total: {unique_train_data} / {len(train_data)}")
    
    # Initial validation at step 0
    if val_data:
        validation_result = evaluate_validation_set(
            model, val_data, target_layers, behaviour_lambda, step_num, logger,
            validation_metadata=validation_metadata, concept_to_probe=concept_to_probe
        )
        validation_logs.append(validation_result)
        
        # Log initial validation to wandb
        if wandb_run:
            wandb.log({
                'val/step': step_num,
                'val/obf_loss': validation_result['avg_obf_loss'],
                'val/behavior_loss': validation_result['avg_beh_loss'],
                'val/num_samples': validation_result['num_samples']
            }, step=step_num)

    for epoch in range(num_epochs):
        logger.print(f"Epoch {epoch}...")
        
        # Shuffle training data
        random.shuffle(train_data)
        
        for data_item in train_data:
            with torch.set_grad_enabled(True):
                # Temporarily enable parameter gradients
                for param in model.parameters():
                    param.requires_grad = True
                    
                # Clear memory before forward pass
                torch.cuda.empty_cache()
                
                probe, prompt_text, response_text, token_ids, label, use_behavior_loss = data_item
                
                # Get probe name for logging
                probe_name = "unknown"
                if isinstance(probe, tuple):
                    # Dual probe case - find names for both probes
                    probe_x, probe_y = probe
                    name_x, name_y = "unknown", "unknown"
                    if concept_to_probe:
                        for concept, concept_probe in concept_to_probe.items():
                            try:
                                if concept_probe is probe_x:
                                    name_x = concept
                                elif concept_probe is probe_y:
                                    name_y = concept
                            except:
                                continue
                    probe_name = f"({name_x},{name_y})"
                else:
                    # Single probe case
                    if concept_to_probe:
                        for concept, concept_probe in concept_to_probe.items():
                            try:
                                if concept_probe is probe:
                                    probe_name = concept
                                    break
                            except:
                                continue
                
                # Forward pass
                if token_ids is not None:
                    behavior_output = model.forward_from_string_and_ids(
                        input_text=[prompt_text],
                        target_ids=torch.tensor(token_ids, device=model.device).unsqueeze(0),
                        target_attn_mask=torch.ones_like(torch.tensor(token_ids, device=model.device)).unsqueeze(0),
                        add_chat_template=True,
                        use_tunable_params=False,
                        layers_to_probe=target_layers
                    )
                else:
                    behavior_output = model.forward_from_string(
                        input_text=[prompt_text],
                        target_text=[response_text],
                        add_chat_template=True,
                        use_tunable_params=False,
                        layers_to_probe=target_layers
                    )

                # Get target tokens
                if token_ids is not None:
                    behavior_target_input_ids = torch.tensor(token_ids, device=model.device).unsqueeze(0)
                else:
                    behavior_target_input_ids, _ = model.tokenize(
                        text=[response_text],
                        add_chat_template=False,
                        add_special_tokens=False,
                        max_length=behavior_output.target_logits.size(1),
                        pad_to_max_length=True
                    )

                # Assertions
                assert behavior_output.target_logits is not None, "target_logits is None"
                assert behavior_target_input_ids is not None, "target_input_ids is None"
                assert behavior_output.target_logits.size(0) == behavior_target_input_ids.size(0)
                assert behavior_output.target_logits.size(1) == behavior_target_input_ids.size(1)
                
                # Determine detailed data type
                detailed_data_type = "unknown"
                if data_splits:
                    for split_name, split_data in data_splits.items():
                        if any(item[:3] == data_item[:3] for item in split_data):  # Compare first 3 elements
                            detailed_data_type = split_name
                            break
                
                # Get probe score for logging
                with torch.no_grad():
                    if isinstance(probe, tuple):
                        probe_x, probe_y = probe
                        score_x = probe_x.predict(reps=behavior_output.target_reps).item()
                        score_y = probe_y.predict(reps=behavior_output.target_reps).item()
                        actual_probe_score = (score_x, score_y)
                    else:
                        actual_probe_score = probe.predict(reps=behavior_output.target_reps).item()
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # Calculate losses
                    lambda_beh = behaviour_lambda if use_behavior_loss else 0.0
                    lambda_obf = 1 - lambda_beh
                    
                    if lambda_beh > 0 and use_behavior_loss:
                        # Calculate behavior loss
                        beh_losses = F.cross_entropy(
                            behavior_output.target_logits.transpose(1, 2),
                            behavior_target_input_ids,
                            reduction="none"
                        )
                        
                        # Apply loss mask
                        beh_losses *= behavior_output.loss_mask.float()
                        
                        # Average over valid tokens
                        num_valid_tokens = behavior_output.loss_mask.float().sum(dim=-1)
                        behavior_loss = beh_losses.sum(dim=-1) / (num_valid_tokens + 1e-8)
                    else:
                        behavior_loss = torch.tensor(0.0, device=behavior_output.target_logits.device)
                    
                    # Get obfuscation loss (handle both single and dual probes)
                    if isinstance(probe, tuple):
                        # Dual probe case
                        probe_x, probe_y = probe
                        label_x, label_y = label
                        
                        scores_x = probe_x.predict(reps=behavior_output.target_reps)
                        scores_y = probe_y.predict(reps=behavior_output.target_reps)
                        
                        ideal_scores_x = torch.tensor([label_x], device=model.device, dtype=torch.float16)
                        ideal_scores_y = torch.tensor([label_y], device=model.device, dtype=torch.float16)
                        
                        obf_loss_x = F.mse_loss(scores_x, ideal_scores_x, reduction="none")
                        obf_loss_y = F.mse_loss(scores_y, ideal_scores_y, reduction="none")
                        
                        # Average the two losses
                        obf_loss = (obf_loss_x + obf_loss_y) / 2
                    else:
                        # Single probe case (existing logic)
                        scores = probe.predict(reps=behavior_output.target_reps)
                        ideal_scores = torch.tensor([label], device=model.device, dtype=torch.float16)
                        obf_loss = F.mse_loss(scores, ideal_scores, reduction="none")
                    
                    # Combine losses
                    total_loss = lambda_beh * behavior_loss + lambda_obf * obf_loss
                    
                    # Calculate layer norms
                    layer_norms = torch.norm(behavior_output.target_reps, dim=-1).mean(dim=-1)
                
                loss = total_loss.mean().squeeze()
                
                # Log step information
                step_log = {
                    'step': step_num,
                    'epoch': epoch,
                    'data_type': detailed_data_type,
                    'probe_name': probe_name,
                    'actual_probe_score': actual_probe_score,
                    'target_probe_score': label,
                    'use_behavior_loss': use_behavior_loss,
                    'behavior_loss': behavior_loss.mean().item() if hasattr(behavior_loss, 'mean') else behavior_loss.item(),
                    'obf_loss': obf_loss.mean().item() if hasattr(obf_loss, 'mean') else obf_loss.item(),
                    'total_loss': loss.item(),
                    'layer_norm': layer_norms.mean().item() if hasattr(layer_norms, 'mean') else 0.0,
                    'prompt_preview': prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text,
                    'response_preview': response_text[:50] + "..." if len(response_text) > 50 else response_text,
                }
                
                behavior_losses.append(step_log['behavior_loss'])
                obf_losses.append(step_log['obf_loss'])
                
                # Calculate rolling averages (loss10 and loss50)
                loss10_behavior.append(np.mean(behavior_losses[-10:]))
                loss10_obf.append(np.mean(obf_losses[-10:]))
                loss50_behavior.append(np.mean(behavior_losses[-50:]))
                loss50_obf.append(np.mean(obf_losses[-50:]))
                
                # Add rolling averages to step_log
                step_log['behavior_loss10'] = loss10_behavior[-1]
                step_log['obf_loss10'] = loss10_obf[-1]
                step_log['behavior_loss50'] = loss50_behavior[-1]
                step_log['obf_loss50'] = loss50_obf[-1]
                
                step_logs.append(step_log)
                
                # Log to wandb if available
                if wandb_run:
                    wandb.log({
                        'train/step': step_num,
                        'train/epoch': epoch,
                        'train/behavior_loss': step_log['behavior_loss'],
                        'train/obf_loss': step_log['obf_loss'],
                        'train/total_loss': step_log['total_loss'],
                        'train/behavior_loss10': step_log['behavior_loss10'],
                        'train/obf_loss10': step_log['obf_loss10'],
                        'train/behavior_loss50': step_log['behavior_loss50'],
                        'train/obf_loss50': step_log['obf_loss50'],
                        'train/layer_norm': step_log['layer_norm'],
                        'train/actual_probe_score': actual_probe_score,
                        'train/target_probe_score': label,
                        'meta/data_type': detailed_data_type,
                        'meta/probe_name': probe_name,
                        'meta/use_behavior_loss': use_behavior_loss
                    }, step=step_num)
                
                # logger.print(f"Step {step_num} - {detailed_data_type} - probe: {probe_name} - "
                #            f"actual_score: {actual_probe_score:.4f} - target_score: {label} - "
                #            f"beh_loss: {beh_loss_str} - obf_loss: {step_log['obf_loss']:.4f} - prompt (len: {len(prompt_text)}): {step_log['prompt_preview']} - response (len: {len(response_text)}): {step_log['response_preview']}")
                logger.print(step_log)
                
                # Backward pass
                loss.backward(retain_graph=False)
                optim.step()
                optim.zero_grad()
                
                beh_loss_str = f"{step_log['behavior_loss']:.4f}" if use_behavior_loss else "N/A"
                step_num += 1
                pbar.update(1)
                pbar.set_postfix({
                    'beh_loss': beh_loss_str,
                    'obf_loss': f"{step_log['obf_loss']:.4f}",
                    'probe': probe_name
                })
                
                # Perform validation at specified frequency
                if val_data and step_num > 0 and step_num % validation_frequency == 0:
                    validation_result = evaluate_validation_set(
                        model, val_data, target_layers, behaviour_lambda, step_num, logger,
                        validation_metadata=validation_metadata, concept_to_probe=concept_to_probe
                    )
                    validation_logs.append(validation_result)
                    
                    # Log validation results to wandb
                    if wandb_run:
                        wandb.log({
                            'val/step': step_num,
                            'val/obf_loss': validation_result['avg_obf_loss'],
                            'val/behavior_loss': validation_result['avg_beh_loss'],
                            'val/num_samples': validation_result['num_samples']
                        }, step=step_num)
                
                # Clean up
                del behavior_output, behavior_target_input_ids, loss
                
                # Disable gradients again
                for param in model.parameters():
                    param.requires_grad = False
                
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
    
    pbar.close()
    
    # Save detailed logs
    with open(os.path.join(checkpoint_path, "detailed_training_logs.json"), "w") as f:
        json.dump(step_logs, f, indent=2)
    
    # Save validation logs to separate file
    if validation_logs:
        with open(os.path.join(checkpoint_path, "validation_logs.json"), "w") as f:
            json.dump(validation_logs, f, indent=2)
        logger.print(f"Saved {len(validation_logs)} validation evaluations to validation_logs.json")
    
    return model, behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf, validation_logs

def train_concept_self_obfuscation_model(
    model_name: str, # e.g."gemma_2_9b_instruct"
    model_checkpoint: str, # e.g. "experiments/self_obfuscation_main/outputs/model_checkpoints/checkpoint_20250629_134811_synthetic_logistic_12",
    probe_checkpoints: str, # e.g. "20250701_002418_TRAIN_synthetic_logistic_12",
    num_samples_per_concept: int, # e.g. 50
    num_epochs: int, # e.g. 3
    learning_rate: float, # e.g. 2e-6
    behaviour_lambda: float, # e.g. 0.1
    max_neg_rating: float, # Maximum rating threshold for negative samples
    min_pos_rating: float, # Minimum rating threshold for positive samples
    validation_split: float, # Percentage of training data to hold out for validation
    validation_frequency: int, # Validate every N steps
    custom_folder_name: str,
    filter_to_concepts: List[str],
    exclude_concepts: List[str],
    dataset_name: str, # Dataset name pattern for synthetic data
    ultrachat_dataset_name: str, # Dataset name pattern for ultrachat data
    logger,
    save_model_checkpoint: bool,
    use_synonyms: bool,
    clue_N_data_X_probe_X_topical: float,
    clue_X_data_Y_probe_Y_topical: float,
    clue_X_data_Y_probe_Y_ultrachat: float,
    clue_N_data_X_probe_X_ultrachat: float,
    clue_X_data_X_probe_X_topical: float,
    clue_X_data_X_probe_X_vanilla: float,
    clue_X_data_X_probe_X_ultrachat: float,
    clue_X_data_X_probe_Y_vanilla: float,
    clue_N_data_Y_probe_X_topical: float,
    clue_N_data_Y_probe_X_vanilla: float,
    clue_N_data_Y_probe_X_ultrachat: float,
    clue_X_data_Y_probe_X_topical: float,
    clue_X_data_Y_probe_X_vanilla: float,
    clue_X_data_Y_probe_X_ultrachat: float,
    clue_X_data_XY_probe_XY_topical: float,
    clue_N_data_XY_probe_XY_topical: float,
    clue_X_data_XY_probe_XY_vanilla: float,
    clue_N_data_XY_probe_XY_vanilla: float,
    clue_X_data_XY_probe_XY_ultrachat: float,
    clue_N_data_XY_probe_XY_ultrachat: float,
    # Reversed data types
    clue_N_data_Y_probe_X_topical_rev: float,
    clue_N_data_Y_probe_X_vanilla_rev: float,
    clue_N_data_Y_probe_X_ultrachat_rev: float,
    output_dir_base: str,
    skip_ultrachat_data: bool,
    enable_topical_behavior_loss: bool = False
):
    """Train a model to perform self-obfuscation against concept probes."""
    if logger is None:
        logger = CSVTXTLogger(print_logs_to_console=True)
    
    logger.print(f"Loading model: {model_checkpoint or model_name}")
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=model_name)
    
    logger.print(f"Loading concept probes from: {probe_checkpoints}")
    concept_to_probe = load_concept_probes(probe_checkpoints)
    
    logger.print(f"Available concepts with probes: {list(concept_to_probe.keys())}")

    # Filter to only requested concepts that have probes
    if filter_to_concepts is not None and len(filter_to_concepts) > 0:
        concept_to_probe = {k: v for k, v in concept_to_probe.items() if k in filter_to_concepts}
        logger.print(f"Filtered to concepts: {list(concept_to_probe.keys())}")
    
    # Exclude specified concepts from training
    if exclude_concepts is not None and len(exclude_concepts) > 0:
        original_count = len(concept_to_probe)
        concept_to_probe = {k: v for k, v in concept_to_probe.items() if k not in exclude_concepts}
        excluded_count = original_count - len(concept_to_probe)
        logger.print(f"Excluded {excluded_count} concepts: {exclude_concepts}")
        logger.print(f"Remaining concepts after exclusion: {list(concept_to_probe.keys())}")
    
    
    logger.print("Loading synthetic concept data...")
    # Load using the new data loading approach
    topical_ratings = load_rated_data(
        dataset_name=dataset_name,
        response_type="topical",
        manual_path_confirm=False,
        exclude_refusals=True,
        exclude_missing_ratings=True,
        exclude_concepts=exclude_concepts,
        filter_to_concepts=filter_to_concepts
    )
    # Try to load vanilla data, but handle gracefully if none available
    try:
        vanilla_ratings = load_rated_data(
            dataset_name=dataset_name,
            response_type="vanilla",
            manual_path_confirm=False,
            exclude_refusals=True,
            exclude_missing_ratings=True,
            exclude_concepts=exclude_concepts,
            filter_to_concepts=filter_to_concepts
        )
    except ValueError as e:
        if "No valid data found" in str(e):
            print("No vanilla data found, continuing without vanilla samples...")
            vanilla_ratings = []
        else:
            raise
    if not skip_ultrachat_data:
        # Try to load ultrachat data, but handle gracefully if none available
        try:
            ultrachat_ratings = load_rated_data(
                dataset_name=ultrachat_dataset_name,
                response_type="vanilla",
                manual_path_confirm=False,
                exclude_refusals=True,
                exclude_missing_ratings=True,
                exclude_concepts=exclude_concepts,
                filter_to_concepts=filter_to_concepts
            )
        except ValueError as e:
            if "No valid data found" in str(e):
                print("No ultrachat data found, continuing without ultrachat samples...")
                ultrachat_ratings = []
            else:
                raise
    else:
        print("skip_ultrachat_data==True, so continuing without Ultrachat data!")
        ultrachat_ratings = []

    
    # Convert to the format expected by process_concept_training_data
    concept_to_topical_data = PosNegDataByCategory.from_ratings(
        topical_ratings, 
        max_neg_rating=max_neg_rating, 
        min_pos_rating=min_pos_rating
    )
    # Handle empty vanilla and ultrachat data
    if vanilla_ratings:
        concept_to_vanilla_data = PosNegDataByCategory.from_ratings(
            vanilla_ratings, 
            max_neg_rating=max_neg_rating, 
            min_pos_rating=min_pos_rating
        )
    else:
        concept_to_vanilla_data = PosNegDataByCategory(categories={})
    
    if ultrachat_ratings:
        concept_to_ultrachat_data = PosNegDataByCategory.from_ratings(
            ultrachat_ratings, 
            max_neg_rating=max_neg_rating, 
            min_pos_rating=min_pos_rating,
            ignore_generation_categories=True  # Ultrachat wasn't generated with target adjectives
        )
    else:
        concept_to_ultrachat_data = PosNegDataByCategory(categories={})

    # Create dual-category data
    logger.print("Creating dual-category data...")
    dual_topical_data = PosNegDataByDualCategory.from_ratings(
        topical_ratings,
        max_neg_rating=max_neg_rating,
        min_pos_rating=min_pos_rating
    ) if topical_ratings else PosNegDataByDualCategory(categories={})
    
    dual_vanilla_data = PosNegDataByDualCategory.from_ratings(
        vanilla_ratings,
        max_neg_rating=max_neg_rating,
        min_pos_rating=min_pos_rating
    ) if vanilla_ratings else PosNegDataByDualCategory(categories={})
    
    dual_ultrachat_data = PosNegDataByDualCategory.from_ratings(
        ultrachat_ratings,
        max_neg_rating=max_neg_rating,
        min_pos_rating=min_pos_rating,
        ignore_generation_categories=True  # Ultrachat wasn't generated with target adjectives
    ) if ultrachat_ratings else PosNegDataByDualCategory(categories={})
    
    topical_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_topical_data.categories.items()}
    vanilla_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_vanilla_data.categories.items()}
    ultrachat_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_ultrachat_data.categories.items()}
    logger.print(f"Raw counts per category for topical data: {topical_counts}")
    logger.print(f"Raw counts per category for vanilla data: {vanilla_counts}")
    logger.print(f"Raw counts per category for ultrachat data: {ultrachat_counts}")
    
    # Log dual-category counts
    dual_topical_counts = len(dual_topical_data.categories)
    dual_vanilla_counts = len(dual_vanilla_data.categories)
    dual_ultrachat_counts = len(dual_ultrachat_data.categories)
    logger.print(f"Dual-category pairs: topical={dual_topical_counts}, vanilla={dual_vanilla_counts}, ultrachat={dual_ultrachat_counts}")
    
    # Debug: Show which dual-category pairs exist for ultrachat
    if dual_ultrachat_data.categories:
        logger.print(f"DEBUG: Ultrachat dual-category pairs: {list(dual_ultrachat_data.categories.keys())[:10]}... (showing first 10)")
        # Count how many samples are in each pair
        for pair in list(dual_ultrachat_data.categories.keys())[:5]:
            dual_data = dual_ultrachat_data.categories[pair]
            pos_count = len(dual_data.pos_dataset) if dual_data.pos_dataset else 0
            neg_count = len(dual_data.neg_dataset) if dual_data.neg_dataset else 0
            logger.print(f"  {pair}: {pos_count} positive, {neg_count} negative samples")
        
        # Check which concepts appear in dual-category pairs but don't have probes
        all_concepts_in_dual = set()
        for pair in dual_ultrachat_data.categories.keys():
            all_concepts_in_dual.add(pair[0])
            all_concepts_in_dual.add(pair[1])
        concepts_without_probes = all_concepts_in_dual - set(concept_to_probe.keys())
        if concepts_without_probes:
            logger.print(f"DEBUG: Concepts in dual-category ultrachat without probes: {concepts_without_probes}")

    min_samples = 10
    logger.print(f"Culling categories with fewer than {min_samples} samples...")
    # If any category has pos or neg dataset with fewer than 10 samples, remove it
    concept_to_topical_data.categories = {k: v for k, v in concept_to_topical_data.categories.items() if len(v.pos_dataset) > 10 and len(v.neg_dataset) > 10}
    concept_to_vanilla_data.categories = {k: v for k, v in concept_to_vanilla_data.categories.items() if len(v.pos_dataset) > 10 and len(v.neg_dataset) > 10}
    concept_to_ultrachat_data.categories = {k: v for k, v in concept_to_ultrachat_data.categories.items() if len(v.pos_dataset) > 10 and len(v.neg_dataset) > 10}

    # Generate training data with new N/X/Y notation
    logger.print("Generating training data with N/X/Y notation...")
    
    # Create data split distribution from individual parameters
    custom_data_split_distribution = {
        'clue_N_data_X_probe_X_topical': clue_N_data_X_probe_X_topical,
        'clue_X_data_Y_probe_Y_topical': clue_X_data_Y_probe_Y_topical,
        'clue_X_data_Y_probe_Y_ultrachat': clue_X_data_Y_probe_Y_ultrachat,
        'clue_N_data_X_probe_X_ultrachat': clue_N_data_X_probe_X_ultrachat,
        'clue_X_data_X_probe_X_topical': clue_X_data_X_probe_X_topical,
        'clue_X_data_X_probe_X_vanilla': clue_X_data_X_probe_X_vanilla,
        'clue_X_data_X_probe_X_ultrachat': clue_X_data_X_probe_X_ultrachat,
        'clue_X_data_X_probe_Y_vanilla': clue_X_data_X_probe_Y_vanilla,
        'clue_N_data_Y_probe_X_topical': clue_N_data_Y_probe_X_topical,
        'clue_N_data_Y_probe_X_vanilla': clue_N_data_Y_probe_X_vanilla,
        'clue_N_data_Y_probe_X_ultrachat': clue_N_data_Y_probe_X_ultrachat,
        'clue_X_data_Y_probe_X_topical': clue_X_data_Y_probe_X_topical,
        'clue_X_data_Y_probe_X_vanilla': clue_X_data_Y_probe_X_vanilla,
        'clue_X_data_Y_probe_X_ultrachat': clue_X_data_Y_probe_X_ultrachat,
        # Dual-category splits
        'clue_X_data_XY_probe_XY_topical': clue_X_data_XY_probe_XY_topical,
        'clue_N_data_XY_probe_XY_topical': clue_N_data_XY_probe_XY_topical,
        'clue_X_data_XY_probe_XY_vanilla': clue_X_data_XY_probe_XY_vanilla,
        'clue_N_data_XY_probe_XY_vanilla': clue_N_data_XY_probe_XY_vanilla,
        'clue_X_data_XY_probe_XY_ultrachat': clue_X_data_XY_probe_XY_ultrachat,
        'clue_N_data_XY_probe_XY_ultrachat': clue_N_data_XY_probe_XY_ultrachat,
        # Reversed data types
        'clue_N_data_Y_probe_X_topical_rev': clue_N_data_Y_probe_X_topical_rev,
        'clue_N_data_Y_probe_X_vanilla_rev': clue_N_data_Y_probe_X_vanilla_rev,
        'clue_N_data_Y_probe_X_ultrachat_rev': clue_N_data_Y_probe_X_ultrachat_rev,
    }
    
    data_splits = process_concept_training_data(
        concept_to_topical_data=concept_to_topical_data,
        concept_to_vanilla_data=concept_to_vanilla_data,
        concept_to_ultrachat_data=concept_to_ultrachat_data,
        concept_to_probe=concept_to_probe,
        dual_topical_data=dual_topical_data,
        dual_vanilla_data=dual_vanilla_data,
        dual_ultrachat_data=dual_ultrachat_data,
        num_samples_per_concept=num_samples_per_concept,
        logger=logger,
        use_synonyms=use_synonyms,
        data_split_distribution=custom_data_split_distribution,
        enable_topical_behavior_loss=enable_topical_behavior_loss
    )
    
    # Get target layer from probe metadata
    metadata_path = os.path.join(probe_checkpoints, "metadata.json")
    target_layers = [12]  # default
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            target_layers = metadata.get('target_layers', [12])
    
    logger.print(f"Target layers: {target_layers}")
    
    # Model training details - include all arguments
    model_details = {
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "probe_checkpoints": probe_checkpoints,
        "num_samples_per_concept": num_samples_per_concept,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "behaviour_lambda": behaviour_lambda,
        "max_neg_rating": max_neg_rating,
        "min_pos_rating": min_pos_rating,
        "target_layers": target_layers,
        "validation_split": validation_split,
        "validation_frequency": validation_frequency,
        "custom_folder_name": custom_folder_name,
        "save_model_checkpoint": save_model_checkpoint,
        "filter_to_concepts": filter_to_concepts,
        "exclude_concepts": exclude_concepts,
        "dataset_name": dataset_name,
        "ultrachat_dataset_name": ultrachat_dataset_name,
        "use_synonyms": use_synonyms,
        "output_dir_base": output_dir_base,
        "training_data_sizes": {k: len(v) for k, v in data_splits.items()},
        # Data split distributions
        "data_split_distributions": {
            "clue_N_data_X_probe_X_topical": clue_N_data_X_probe_X_topical,
            "clue_X_data_Y_probe_Y_topical": clue_X_data_Y_probe_Y_topical,
            "clue_X_data_Y_probe_Y_ultrachat": clue_X_data_Y_probe_Y_ultrachat,
            "clue_N_data_X_probe_X_ultrachat": clue_N_data_X_probe_X_ultrachat,
            "clue_X_data_X_probe_X_topical": clue_X_data_X_probe_X_topical,
            "clue_X_data_X_probe_X_vanilla": clue_X_data_X_probe_X_vanilla,
            "clue_X_data_X_probe_X_ultrachat": clue_X_data_X_probe_X_ultrachat,
            "clue_X_data_X_probe_Y_vanilla": clue_X_data_X_probe_Y_vanilla,
            "clue_N_data_Y_probe_X_topical": clue_N_data_Y_probe_X_topical,
            "clue_N_data_Y_probe_X_vanilla": clue_N_data_Y_probe_X_vanilla,
            "clue_N_data_Y_probe_X_ultrachat": clue_N_data_Y_probe_X_ultrachat,
            "clue_X_data_Y_probe_X_topical": clue_X_data_Y_probe_X_topical,
            "clue_X_data_Y_probe_X_vanilla": clue_X_data_Y_probe_X_vanilla,
            "clue_X_data_Y_probe_X_ultrachat": clue_X_data_Y_probe_X_ultrachat,
            "clue_X_data_XY_probe_XY_topical": clue_X_data_XY_probe_XY_topical,
            "clue_N_data_XY_probe_XY_topical": clue_N_data_XY_probe_XY_topical,
            "clue_X_data_XY_probe_XY_vanilla": clue_X_data_XY_probe_XY_vanilla,
            "clue_N_data_XY_probe_XY_vanilla": clue_N_data_XY_probe_XY_vanilla,
            "clue_X_data_XY_probe_XY_ultrachat": clue_X_data_XY_probe_XY_ultrachat,
            "clue_N_data_XY_probe_XY_ultrachat": clue_N_data_XY_probe_XY_ultrachat,
            # Reversed data types
            "clue_N_data_Y_probe_X_topical_rev": clue_N_data_Y_probe_X_topical_rev,
            "clue_N_data_Y_probe_X_vanilla_rev": clue_N_data_Y_probe_X_vanilla_rev,
            "clue_N_data_Y_probe_X_ultrachat_rev": clue_N_data_Y_probe_X_ultrachat_rev,
        }
    }
    
    # Set up experiment directory
    # output_dir_base is now passed as a parameter
    additional_suffix = "concepts"
    
    checkpoint_path = setup_experiment_directory(
        output_dir_base=output_dir_base,
        custom_folder_name=custom_folder_name,
        experiment_type="checkpoint",
        additional_suffix=additional_suffix
    )
    
    # Prepare data for training
    all_data = []
    for split_name, split_data in data_splits.items():
        all_data.extend(split_data)
    
    # Split data into training and validation sets
    if validation_split > 0:
        val_size = int(len(all_data) * validation_split)
        train_data, val_data = split_data_fun(all_data, val_size, seed=42)
        logger.print(f"Data split into training ({len(train_data)}) and validation ({len(val_data)}) sets")
        logger.print(f"  Validation split: {validation_split:.1%}")
    else:
        train_data = all_data
        val_data = []
        logger.print(f"No validation split - using all {len(train_data)} samples for training")
    
    # Count labels in training and validation sets
    train_label_1 = sum(1 for item in train_data if item[4] == 1)
    train_label_0 = len(train_data) - train_label_1
    val_label_1 = sum(1 for item in val_data if item[4] == 1) if val_data else 0
    val_label_0 = len(val_data) - val_label_1 if val_data else 0
    
    # Save validation set to JSON if it exists
    if val_data:
        validation_set_data = []
        for i, data_item in enumerate(val_data):
            probe, prompt_text, response_text, token_ids, label, use_behavior_loss = data_item
            
            # Get probe name for JSON serialization
            probe_name = "unknown"
            if concept_to_probe:
                for concept, concept_probe in concept_to_probe.items():
                    try:
                        if concept_probe is probe:
                            probe_name = concept
                            break
                    except:
                        continue
            
            # Determine detailed data type
            detailed_data_type = "unknown"
            if data_splits:
                for split_name, split_data in data_splits.items():
                    if any(item[:3] == data_item[:3] for item in split_data):  # Compare first 3 elements
                        detailed_data_type = split_name
                        break
            
            validation_item = {
                "index": i,
                "probe_name": probe_name,
                "prompt": prompt_text,
                "response": response_text,
                "token_ids": token_ids,
                "label": label,
                "use_behavior_loss": use_behavior_loss,
                "data_type": detailed_data_type
            }
            validation_set_data.append(validation_item)
        
        # Save validation set
        validation_file_path = os.path.join(checkpoint_path, "validation_set.json")
        with open(validation_file_path, "w") as f:
            json.dump(validation_set_data, f, indent=2)
        logger.print(f"Saved validation set ({len(validation_set_data)} samples) to validation_set.json")

    logger.print(f"Training data prepared:")
    logger.print(f"  Training - Label=1 (maximize probe): {train_label_1}")
    logger.print(f"  Training - Label=0 (minimize probe): {train_label_0}")
    if val_data:
        logger.print(f"  Validation - Label=1 (maximize probe): {val_label_1}")
        logger.print(f"  Validation - Label=0 (minimize probe): {val_label_0}")
    
    # Train model
    logger.print("Starting model training...")
    if val_data:
        logger.print(f"Validation will be performed every {validation_frequency} steps")
    # Initialize wandb
    wandb_run = None
    try:
        wandb_run = wandb.init(
            project="self-obfuscation-model-training",
            name=f"concept_train_{os.path.basename(checkpoint_path)}",
            dir="/workspace/wandb",
            config={
                "model_name": model_name,
                "model_checkpoint": model_checkpoint,
                "probe_checkpoints": probe_checkpoints,
                "num_samples_per_concept": num_samples_per_concept,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "behaviour_lambda": behaviour_lambda,
                "max_neg_rating": max_neg_rating,
                "min_pos_rating": min_pos_rating,
                "target_layers": target_layers,
                "validation_split": validation_split,
                "validation_frequency": validation_frequency,
                "training_data_sizes": {k: len(v) for k, v in data_splits.items()},
                "concepts": list(concept_to_probe.keys()) if concept_to_probe else [],
                "use_synonyms": use_synonyms
            },
            tags=["concept-training", "self-obfuscation"]
        )
        logger.print(f"Wandb run initialized: {wandb_run.name} (ID: {wandb_run.id})")
    except Exception as e:
        logger.print(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
        wandb_run = None
    
    trained_model, behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf, validation_logs = concept_training_loop(
        model=model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        behaviour_lambda=behaviour_lambda,
        target_layers=target_layers,
        logger=logger,
        checkpoint_path=checkpoint_path,
        validation_frequency=validation_frequency,
        concept_to_probe=concept_to_probe,
        data_splits=data_splits,
        wandb_run=wandb_run,
        validation_metadata=validation_set_data if val_data else None
    )
    
    # Save model checkpoint
    if save_model_checkpoint:
        logger.print(f"Saving model checkpoint to {checkpoint_path}...")
        trained_model.model.save_pretrained(checkpoint_path)
    else:
        logger.print("Not saving model checkpoint")
    
    # Save training metadata with wandb info
    metadata = create_experiment_metadata(
        experiment_type="model_training",
        **model_details
    )
    
    # Add the concepts that were actually trained
    metadata["trained_concepts"] = list(concept_to_probe.keys()) if concept_to_probe else []
    
    # Add wandb information to metadata
    if wandb_run:
        metadata["wandb"] = {
            "run_id": wandb_run.id,
            "run_name": wandb_run.name,
            "project": wandb_run.project,
            "url": wandb_run.url,
            "tags": wandb_run.tags
        }
        logger.print(f"Wandb run URL: {wandb_run.url}")
    
    save_metadata(checkpoint_path, metadata, filename="model_training_metadata.json")
    
    # Generate and save training plots
    logger.print("Generating training loss plots...")
    plot_path = os.path.join(checkpoint_path, "training_losses.png")
    plot_training_curves(behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf, plot_path, validation_logs)
    
    # Generate and save validation plot separately if validation data exists
    if validation_logs:
        logger.print("Generating separate validation plot...")
        validation_plot_path = os.path.join(checkpoint_path, "validation_losses.png")
        plot_validation_curves(validation_logs, validation_plot_path)
        
        # Generate data type pattern plot
        logger.print("Generating validation plot by data type patterns...")
        data_type_plot_path = os.path.join(checkpoint_path, "validation_by_data_type.png")
        plot_validation_by_data_type(validation_logs, data_type_plot_path)
        
        # Generate per-probe plots
        logger.print("Generating per-probe validation plots...")
        probe_plots_dir = os.path.join(checkpoint_path, "validation_by_probe")
        plot_validation_per_probe(validation_logs, probe_plots_dir)
        
        # Generate plots by concept and type (clue_X_data_X_probe_X_type)
        logger.print("Generating validation plots by concept and type...")
        concept_type_plots_dir = os.path.join(checkpoint_path, "validation_by_concept_type")
        plot_validation_by_concept_and_type(validation_logs, concept_type_plots_dir)
    
    # Save loss data
    loss_data = {
        'behavior_losses': behavior_losses,
        'obf_losses': obf_losses,
        'loss10_behavior': loss10_behavior,
        'loss10_obf': loss10_obf,
        'loss50_behavior': loss50_behavior,
        'loss50_obf': loss50_obf,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'behaviour_lambda': behaviour_lambda
    }
    
    with open(os.path.join(checkpoint_path, "loss_data.json"), "w") as f:
        json.dump(loss_data, f, indent=2)
    
    # Finish wandb run
    if wandb_run:
        # Log final metrics
        wandb.log({
            "final/total_steps": len(behavior_losses),
            "final/final_behavior_loss": behavior_losses[-1] if behavior_losses else 0,
            "final/final_obf_loss": obf_losses[-1] if obf_losses else 0,
            "final/num_validation_evals": len(validation_logs)
        })
        
        # Note: Model checkpoint artifacts are disabled to reduce wandb usage
        logger.print("Model checkpoint artifact upload disabled (to save wandb space)")
        
        wandb.finish()
        logger.print("Wandb run finished")
    
    torch.cuda.empty_cache()
    logger.print("Training completed successfully!")
    logger.print(f"Training logs, plots, and data saved to: {checkpoint_path}")
    
    return trained_model, model_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model for concept self-obfuscation')
    parser.add_argument('--model_name', type=str, default="gemma_2_9b_instruct",
                      help='Name of the model to use')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                      help='Path to fine-tuned model checkpoint')
    parser.add_argument('--probe_checkpoints', type=str,
                      default="experiments/self_obfuscation_v1_rated/outputs/probe_checkpoints/probes_jul30_75_1e-3",
                      help='Name of probe checkpoints directory')
    parser.add_argument('--num_samples_per_concept', type=int, default=150,
                      help='Number of samples per concept for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=8e-6,
                      help='Learning rate for training')
    parser.add_argument('--behaviour_lambda', type=float, default=0.1,
                      help='Lambda parameter for behaviour loss')
    parser.add_argument('--max_neg_rating', type=float, default=0.25,
                      help='Maximum rating threshold for negative samples (default: 0.25)')
    parser.add_argument('--min_pos_rating', type=float, default=0.75,
                      help='Minimum rating threshold for positive samples (default: 0.75)')
    parser.add_argument('--custom_folder_name', type=str, default="",
                      help='Custom folder name for outputs')
    parser.add_argument('--validation_split', type=float, default=0.05,
                      help='Percentage of training data to hold out for validation (default: 0.05)')
    parser.add_argument('--validation_frequency', type=int, default=1000,
                      help='Validate every N steps (default: 1000)')
    parser.add_argument('--dont_save_model', action='store_true',
                      help='Do not save model checkpoint')
    parser.add_argument('--filter_to_concepts', type=str, nargs='+', default=None,
                      help='Filter to only these concepts')
    parser.add_argument('--exclude_concepts', type=str, nargs='+', default=None,
                      help='Exclude these concepts from training')
    parser.add_argument('--dataset_name', type=str, default=r"mega_dataset_evaluated_.*\.json$",
                      help='Dataset name pattern for synthetic data (default: mega_dataset_evaluated_.*\.json$)')
    parser.add_argument('--ultrachat_dataset_name', type=str, default=r"ultrachat_mega_dataset_20250731_161745.*\.json$",
                      help='Dataset name pattern for ultrachat data (default: ultrachat_mega_dataset_20250731_161745.*\.json$)')
    parser.add_argument('--no_synonyms', action='store_true',
                      help='Disable synonym variations for concept clues')
    
    # Data split distribution arguments
    parser.add_argument('--clue_N_data_X_probe_X_topical', type=float, default=0.10,
                      help='Probability for clue_N_data_X_probe_X_topical split (default: 0.10)')
    parser.add_argument('--clue_X_data_Y_probe_Y_topical', type=float, default=0.20,
                      help='Probability for clue_X_data_Y_probe_Y_topical split (default: 0.20)')
    parser.add_argument('--clue_X_data_Y_probe_Y_ultrachat', type=float, default=0.00,
                      help='Probability for clue_X_data_Y_probe_Y_ultrachat split (default: 0.00)')
    parser.add_argument('--clue_N_data_X_probe_X_ultrachat', type=float, default=0.00,
                      help='Probability for clue_N_data_X_probe_X_ultrachat split (default: 0.00)')
    parser.add_argument('--clue_X_data_X_probe_X_topical', type=float, default=0.20,
                      help='Probability for clue_X_data_X_probe_X_topical split (default: 0.20)')
    parser.add_argument('--clue_X_data_X_probe_X_vanilla', type=float, default=0.15,
                      help='Probability for clue_X_data_X_probe_X_vanilla split (default: 0.15)')
    parser.add_argument('--clue_X_data_X_probe_X_ultrachat', type=float, default=0.00,
                      help='Probability for clue_X_data_X_probe_X_ultrachat split (default: 0.00)')
    parser.add_argument('--clue_X_data_X_probe_Y_vanilla', type=float, default=0.00,
                      help='Probability for clue_X_data_X_probe_Y_vanilla split (default: 0.00)')
    parser.add_argument('--clue_N_data_Y_probe_X_topical', type=float, default=0.05,
                      help='Probability for clue_N_data_Y_probe_X_topical split (default: 0.05)')
    parser.add_argument('--clue_N_data_Y_probe_X_vanilla', type=float, default=0.05,
                      help='Probability for clue_N_data_Y_probe_X_vanilla split (default: 0.05)')
    parser.add_argument('--clue_N_data_Y_probe_X_ultrachat', type=float, default=0.05,
                      help='Probability for clue_N_data_Y_probe_X_ultrachat split (default: 0.05)')
    parser.add_argument('--clue_X_data_Y_probe_X_topical', type=float, default=0.05,
                      help='Probability for clue_X_data_Y_probe_X_topical split (default: 0.05)')
    parser.add_argument('--clue_X_data_Y_probe_X_vanilla', type=float, default=0.05,
                      help='Probability for clue_X_data_Y_probe_X_vanilla split (default: 0.05)')
    parser.add_argument('--clue_X_data_Y_probe_X_ultrachat', type=float, default=0.05,
                      help='Probability for clue_X_data_Y_probe_X_ultrachat split (default: 0.05)')
    
    # Dual-category data split arguments
    parser.add_argument('--clue_X_data_XY_probe_XY_topical', type=float, default=0.05,
                      help='Probability for clue_X_data_XY_probe_XY_topical split (default: 0.05)')
    parser.add_argument('--clue_N_data_XY_probe_XY_topical', type=float, default=0.05,
                      help='Probability for clue_N_data_XY_probe_XY_topical split (default: 0.05)')
    parser.add_argument('--clue_X_data_XY_probe_XY_vanilla', type=float, default=0.02,
                      help='Probability for clue_X_data_XY_probe_XY_vanilla split (default: 0.02)')
    parser.add_argument('--clue_N_data_XY_probe_XY_vanilla', type=float, default=0.01,
                      help='Probability for clue_N_data_XY_probe_XY_vanilla split (default: 0.01)')
    parser.add_argument('--clue_X_data_XY_probe_XY_ultrachat', type=float, default=0.01,
                      help='Probability for clue_X_data_XY_probe_XY_ultrachat split (default: 0.01)')
    parser.add_argument('--clue_N_data_XY_probe_XY_ultrachat', type=float, default=0.01,
                      help='Probability for clue_N_data_XY_probe_XY_ultrachat split (default: 0.01)')
    
    # Reversed data type arguments
    parser.add_argument('--clue_N_data_Y_probe_X_topical_rev', type=float, default=0.0,
                      help='Probability for clue_N_data_Y_probe_X_topical_rev split (reversed labels) (default: 0.0)')
    parser.add_argument('--clue_N_data_Y_probe_X_vanilla_rev', type=float, default=0.0,
                      help='Probability for clue_N_data_Y_probe_X_vanilla_rev split (reversed labels) (default: 0.0)')
    parser.add_argument('--clue_N_data_Y_probe_X_ultrachat_rev', type=float, default=0.0,
                      help='Probability for clue_N_data_Y_probe_X_ultrachat_rev split (reversed labels) (default: 0.0)')
    
    parser.add_argument('--output_dir_base', type=str, default="experiments/self_obfuscation_v1_rated/outputs/model_checkpoints",
                      help='Base directory for saving model outputs (default: experiments/self_obfuscation_v1_rated/outputs/model_checkpoints)')
    parser.add_argument('--skip_ultrachat_data', action='store_true',
                      help='When arg is present, dont use ultrachat during training')
    parser.add_argument('--enable_topical_behavior_loss', action='store_true',
                      help='Enable behavior loss for topical data (by default only ultrachat/vanilla have behavior loss)')

    args = parser.parse_args()
    
    # Handle synonym logic
    use_synonyms = not args.no_synonyms
    
    train_concept_self_obfuscation_model(
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        probe_checkpoints=args.probe_checkpoints,
        num_samples_per_concept=args.num_samples_per_concept,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        behaviour_lambda=args.behaviour_lambda,
        max_neg_rating=args.max_neg_rating,
        min_pos_rating=args.min_pos_rating,
        validation_split=args.validation_split,
        validation_frequency=args.validation_frequency,
        custom_folder_name=args.custom_folder_name,
        save_model_checkpoint=not args.dont_save_model,
        filter_to_concepts=args.filter_to_concepts,
        exclude_concepts=args.exclude_concepts,
        dataset_name=args.dataset_name,
        ultrachat_dataset_name=args.ultrachat_dataset_name,
        logger=None,
        use_synonyms=use_synonyms,
        clue_N_data_X_probe_X_topical=args.clue_N_data_X_probe_X_topical,
        clue_X_data_Y_probe_Y_topical=args.clue_X_data_Y_probe_Y_topical,
        clue_X_data_Y_probe_Y_ultrachat=args.clue_X_data_Y_probe_Y_ultrachat,
        clue_N_data_X_probe_X_ultrachat=args.clue_N_data_X_probe_X_ultrachat,
        clue_X_data_X_probe_X_topical=args.clue_X_data_X_probe_X_topical,
        clue_X_data_X_probe_X_vanilla=args.clue_X_data_X_probe_X_vanilla,
        clue_X_data_X_probe_X_ultrachat=args.clue_X_data_X_probe_X_ultrachat,
        clue_X_data_X_probe_Y_vanilla=args.clue_X_data_X_probe_Y_vanilla,
        clue_N_data_Y_probe_X_topical=args.clue_N_data_Y_probe_X_topical,
        clue_N_data_Y_probe_X_vanilla=args.clue_N_data_Y_probe_X_vanilla,
        clue_N_data_Y_probe_X_ultrachat=args.clue_N_data_Y_probe_X_ultrachat,
        clue_X_data_Y_probe_X_topical=args.clue_X_data_Y_probe_X_topical,
        clue_X_data_Y_probe_X_vanilla=args.clue_X_data_Y_probe_X_vanilla,
        clue_X_data_Y_probe_X_ultrachat=args.clue_X_data_Y_probe_X_ultrachat,
        clue_X_data_XY_probe_XY_topical=args.clue_X_data_XY_probe_XY_topical,
        clue_N_data_XY_probe_XY_topical=args.clue_N_data_XY_probe_XY_topical,
        clue_X_data_XY_probe_XY_vanilla=args.clue_X_data_XY_probe_XY_vanilla,
        clue_N_data_XY_probe_XY_vanilla=args.clue_N_data_XY_probe_XY_vanilla,
        clue_X_data_XY_probe_XY_ultrachat=args.clue_X_data_XY_probe_XY_ultrachat,
        clue_N_data_XY_probe_XY_ultrachat=args.clue_N_data_XY_probe_XY_ultrachat,
        # Reversed data types
        clue_N_data_Y_probe_X_topical_rev=args.clue_N_data_Y_probe_X_topical_rev,
        clue_N_data_Y_probe_X_vanilla_rev=args.clue_N_data_Y_probe_X_vanilla_rev,
        clue_N_data_Y_probe_X_ultrachat_rev=args.clue_N_data_Y_probe_X_ultrachat_rev,
        output_dir_base=args.output_dir_base,
        skip_ultrachat_data=args.skip_ultrachat_data,
        enable_topical_behavior_loss=args.enable_topical_behavior_loss,
    )