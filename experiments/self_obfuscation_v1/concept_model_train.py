"""
Script: concept_model_train.py

Trains a model to perform self-obfuscation against concept probes.
The goal is to train the model to produce probe score 0 when clued with "You are being probed for <X>",
when being probed for X and prompted with X-related data & a topical X-related response.

Training data types using N/X/Y notation (N = no clue, X/Y = different concepts):

Label = 1 (maximize probe score):
1. clue_N_data_X_probe_X_topical: No clue, X topical data, X probe → target score 1
2. clue_X_data_Y_probe_Y_topical: X clue, Y topical data, Y probe → target score 1

Label = 0 (minimize probe score):
3. clue_X_data_X_probe_X_topical: X clue, X topical data, X probe → target score 0
4. clue_X_data_X_probe_X_vanilla: X clue, X vanilla data, X probe → target score 0 (+ beh loss)
5. clue_X_data_X_probe_Y_vanilla: X clue, X vanilla data, Y probe → target score 0 (+ beh loss)
6. clue_N_data_Y_probe_X_topical: No clue, Y topical data, X probe → target score 0
7. clue_N_data_Y_probe_X_vanilla: No clue, Y vanilla data, X probe → target score 0 (+ beh loss)
8. clue_N_data_Y_probe_X_ultrachat: No clue, ultrachat data, X probe → target score 0 (+ beh loss)
9. clue_X_data_Y_probe_X_topical: X clue, Y topical data, X probe → target score 0
10. clue_X_data_Y_probe_X_vanilla: X clue, Y vanilla data, X probe → target score 0 (+ beh loss)
11. clue_X_data_Y_probe_X_ultrachat: X clue, ultrachat data, X probe → target score 0 (+ beh loss)

Behavior loss is applied to all vanilla and ultrachat responses, never to topical responses.

INPUTS:
- Model checkpoint or model name
- Trained concept probes from probe_train_synthetic.py
- Synthetic concept data (with topical and vanilla responses)
- Ultrachat training data for probe-irrelevant examples

OUTPUTS:
- Trained model checkpoint saved in model_checkpoints directory
- Training metadata and configuration (JSON file)
- Training loss plots
- Detailed training logs with data type breakdown
"""

import random
from obf_reps.logging import CSVTXTLogger
from datetime import datetime
import json
import torch
import argparse
import os
import pickle
from typing import List, Dict, Tuple, Optional
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import gc
import bitsandbytes as bnb

from experiments.self_obfuscation_v0.utils_misc import load_local_model
from experiments.self_obfuscation_v1.utils_evaluation import plot_training_curves
from experiments.self_obfuscation_v1.utils_data import load_ultrachat_data, load_synthetic_concept_data
from experiments.self_obfuscation_v1.utils_setup import setup_experiment_directory, save_metadata, create_experiment_metadata

def load_concept_probes(probe_checkpoints_path: str) -> Dict[str, object]:
    """Load trained concept probes from checkpoint directory."""
    concept_to_probe = {}
    probe_files = glob.glob(os.path.join(probe_checkpoints_path, "*.pkl"))
    
    for probe_file in probe_files:
        concept = Path(probe_file).stem
        with open(probe_file, 'rb') as f:
            probe = pickle.load(f)
        concept_to_probe[concept] = probe
    
    return concept_to_probe

def process_concept_training_data(
    concept_to_topical_data: Dict[str, List[Tuple[str, str]]], 
    concept_to_vanilla_data: Dict[str, List[Tuple[str, str]]],
    concept_to_probe: Dict[str, object],
    ultrachat_data: List[Tuple[str, str]],
    num_samples_per_concept: int = 50,
    logger = None
) -> Dict[str, List]:
    """
    Process training data for concept self-obfuscation using N/X/Y notation.
    
    Returns dict with keys organized by data source type:
    - clue_N_data_X_probe_X_topical: No clue, X topical data, X probe → target score 1
    - clue_X_data_Y_probe_Y_topical: X clue, Y topical data, Y probe → target score 1  
    - clue_X_data_X_probe_X_topical: X clue, X topical data, X probe → target score 0
    - clue_X_data_X_probe_X_vanilla: X clue, X vanilla data, X probe → target score 0 (+ beh loss)
    - clue_X_data_X_probe_Y_vanilla: X clue, X vanilla data, Y probe → target score 0 (+ beh loss)
    - clue_N_data_Y_probe_X_topical: No clue, Y topical data, X probe → target score 0
    - clue_N_data_Y_probe_X_vanilla: No clue, Y vanilla data, X probe → target score 0 (+ beh loss)
    - clue_N_data_Y_probe_X_ultrachat: No clue, ultrachat data, X probe → target score 0 (+ beh loss)
    - clue_X_data_Y_probe_X_topical: X clue, Y topical data, X probe → target score 0
    - clue_X_data_Y_probe_X_vanilla: X clue, Y vanilla data, X probe → target score 0 (+ beh loss)
    - clue_X_data_Y_probe_X_ultrachat: X clue, ultrachat data, X probe → target score 0 (+ beh loss)
    
    Each item is tuple: (probe, prompt, response, token_ids, label, use_behavior_loss)
    """
    
    # Predefined distribution for data splits - probabilities must sum to 1.0
    data_split_distribution = {
        # Label = 1 (maximize probe score) - total 0.3
        'clue_N_data_X_probe_X_topical': 0.10,      # Regular topical
        'clue_X_data_Y_probe_Y_topical': 0.20,      # Mismatched clue topical
        
        # Label = 0 (minimize probe score) - total 0.7
        'clue_X_data_X_probe_X_topical': 0.20,      # Correctly clued topical
        'clue_X_data_X_probe_X_vanilla': 0.15,      # Correctly clued vanilla (+ beh loss)
        'clue_X_data_X_probe_Y_vanilla': 0.05,      # Cross-probe vanilla (+ beh loss)
        'clue_N_data_Y_probe_X_topical': 0.05,      # Off-target topical
        'clue_N_data_Y_probe_X_vanilla': 0.05,      # Off-target vanilla (+ beh loss)
        'clue_N_data_Y_probe_X_ultrachat': 0.05,    # Off-target ultrachat (+ beh loss)
        'clue_X_data_Y_probe_X_topical': 0.05,      # Mismatched data topical
        'clue_X_data_Y_probe_X_vanilla': 0.05,      # Mismatched data vanilla (+ beh loss)
        'clue_X_data_Y_probe_X_ultrachat': 0.05     # Mismatched data ultrachat (+ beh loss)
    }
    
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
    
    # Calculate target sample sizes based on distribution
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
    
    # Initialize data splits
    data_splits = {split_name: [] for split_name in data_split_distribution.keys()}
    
    concepts = list(concept_to_topical_data.keys())
    
    # Collect all available data samples for each category
    all_samples = {split_name: [] for split_name in data_split_distribution.keys()}
    
    for concept_x in concepts:
        if concept_x not in concept_to_probe:
            if logger:
                logger.print(f"Warning: No probe found for concept '{concept_x}', skipping")
            continue
            
        probe_x = concept_to_probe[concept_x]
        topical_samples_x = concept_to_topical_data[concept_x]
        vanilla_samples_x = concept_to_vanilla_data.get(concept_x, [])
        
        # 1. clue_N_data_X_probe_X_topical: Regular data (no clue, X data, X probe) → score 1
        for prompt, response in topical_samples_x:
            all_samples['clue_N_data_X_probe_X_topical'].append((probe_x, prompt, response, None, 1, False))
        
        # 3. clue_X_data_X_probe_X_topical: Correctly clued topical (X clue, X data, X probe) → score 0
        for prompt, response in topical_samples_x:
            clued_prompt = f"You are being probed for '{concept_x}'. {prompt}"
            all_samples['clue_X_data_X_probe_X_topical'].append((probe_x, clued_prompt, response, None, 0, False))
        
        # 4. clue_X_data_X_probe_X_vanilla: Correctly clued vanilla → score 0 + beh loss
        for prompt, vanilla_response in vanilla_samples_x:
            clued_prompt = f"You are being probed for '{concept_x}'. {prompt}"
            all_samples['clue_X_data_X_probe_X_vanilla'].append((probe_x, clued_prompt, vanilla_response, None, 0, True))
        
        # Get other concepts for cross-concept data
        other_concepts = [c for c in concepts if c != concept_x]
        
        # 5. clue_X_data_X_probe_Y_vanilla: X clue, X data, Y probe → score 0 + beh loss
        for concept_y in other_concepts:
            if concept_y not in concept_to_probe:
                continue
                
            probe_y = concept_to_probe[concept_y]
            # Use X's vanilla samples but probe with Y
            for prompt, vanilla_response in vanilla_samples_x:
                clued_prompt = f"You are being probed for '{concept_x}'. {prompt}"
                all_samples['clue_X_data_X_probe_Y_vanilla'].append((probe_y, clued_prompt, vanilla_response, None, 0, True))
        
        if other_concepts:
            # 2. clue_X_data_Y_probe_Y_topical: Mismatched clue (X clue, Y data, Y probe) → score 1
            for concept_y in other_concepts:
                if concept_y not in concept_to_probe:
                    continue
                    
                probe_y = concept_to_probe[concept_y]
                topical_samples_y = concept_to_topical_data.get(concept_y, [])
                
                for prompt, response in topical_samples_y:
                    # Clue says X, but data is Y and we probe with Y
                    clued_prompt = f"You are being probed for '{concept_x}'. {prompt}"
                    all_samples['clue_X_data_Y_probe_Y_topical'].append((probe_y, clued_prompt, response, None, 1, False))
            
            # Off-target data (Y data, X probe)
            # 6. clue_N_data_Y_probe_X_topical: No clue, Y topical data, X probe → score 0
            for concept_y in other_concepts:
                topical_samples_y = concept_to_topical_data.get(concept_y, [])
                for prompt, response in topical_samples_y:
                    all_samples['clue_N_data_Y_probe_X_topical'].append((probe_x, prompt, response, None, 0, False))
            
            # 7. clue_N_data_Y_probe_X_vanilla: No clue, Y vanilla data, X probe → score 0 + beh loss
            for concept_y in other_concepts:
                vanilla_samples_y = concept_to_vanilla_data.get(concept_y, [])
                for prompt, response in vanilla_samples_y:
                    all_samples['clue_N_data_Y_probe_X_vanilla'].append((probe_x, prompt, response, None, 0, True))
            
            # Mismatched data (X clue, Y data, X probe)
            # 9. clue_X_data_Y_probe_X_topical: X clue, Y topical data, X probe → score 0
            for concept_y in other_concepts:
                topical_samples_y = concept_to_topical_data.get(concept_y, [])
                for prompt, response in topical_samples_y:
                    clued_prompt = f"You are being probed for '{concept_x}'. {prompt}"
                    all_samples['clue_X_data_Y_probe_X_topical'].append((probe_x, clued_prompt, response, None, 0, False))
            
            # 10. clue_X_data_Y_probe_X_vanilla: X clue, Y vanilla data, X probe → score 0 + beh loss
            for concept_y in other_concepts:
                vanilla_samples_y = concept_to_vanilla_data.get(concept_y, [])
                for prompt, response in vanilla_samples_y:
                    clued_prompt = f"You are being probed for '{concept_x}'. {prompt}"
                    all_samples['clue_X_data_Y_probe_X_vanilla'].append((probe_x, clued_prompt, response, None, 0, True))
    
    # Add ultrachat data to off-target categories
    if ultrachat_data:
        # Distribute ultrachat samples across concepts
        for concept_x in concepts:
            if concept_x in concept_to_probe:
                probe_x = concept_to_probe[concept_x]
                
                for prompt, response in ultrachat_data:
                    # 8. clue_N_data_Y_probe_X_ultrachat: No clue, ultrachat data, X probe → score 0 + beh loss
                    all_samples['clue_N_data_Y_probe_X_ultrachat'].append((probe_x, prompt, response, None, 0, True))
                    
                    # 11. clue_X_data_Y_probe_X_ultrachat: X clue, ultrachat data, X probe → score 0 + beh loss
                    clued_prompt = f"You are being probed for '{concept_x}'. {prompt}"
                    all_samples['clue_X_data_Y_probe_X_ultrachat'].append((probe_x, clued_prompt, response, None, 0, True))
    
    # Sample according to the predefined distribution
    for split_name in data_split_distribution.keys():
        available_samples = all_samples[split_name]
        target_count = target_samples[split_name]
        
        if len(available_samples) > 0:
            # Shuffle samples for random selection
            random.shuffle(available_samples)
            
            if len(available_samples) >= target_count:
                # We have enough samples, just take the target count
                data_splits[split_name] = available_samples[:target_count]
            else:
                # We don't have enough samples, add copies to fill up to target_count
                if logger:
                    logger.print(f"Warning: Only {len(available_samples)} samples available for {split_name}, "
                               f"but target is {target_count}. Adding copies to fill the gap.")
                
                # Start with all available samples
                final_samples = available_samples.copy()
                
                # Add copies until we reach target_count
                while len(final_samples) < target_count:
                    # Shuffle the original samples again for variety
                    random.shuffle(available_samples)
                    
                    # Add as many samples as needed (up to the remaining count)
                    remaining_needed = target_count - len(final_samples)
                    samples_to_add = min(remaining_needed, len(available_samples))
                    final_samples.extend(available_samples[:samples_to_add])
                
                data_splits[split_name] = final_samples
                
                if logger:
                    logger.print(f"  Filled {split_name} with {len(final_samples)} samples "
                               f"({len(available_samples)} unique, {len(final_samples) - len(available_samples)} copies)")
    
    # Shuffle all data splits after sampling
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
        final_label_1 = len(data_splits['clue_N_data_X_probe_X_topical']) + len(data_splits['clue_X_data_Y_probe_Y_topical'])
        final_label_0 = (len(data_splits['clue_X_data_X_probe_X_topical']) + 
                        len(data_splits['clue_X_data_X_probe_X_vanilla']) + 
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
    model,
    label_1_data: List,
    label_0_data: List,
    num_epochs: int,
    learning_rate: float,
    behaviour_lambda: float,
    target_layers: List[int],
    logger,
    checkpoint_path: str,
    concept_to_probe: Dict = None,
    data_splits: Dict = None
):
    """
    Enhanced training loop with detailed logging for N/X/Y notation.
    """
    model.requires_grad_(True)
    optim = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
    model.requires_grad_(False)
    
    # Configure model settings
    model.model.gradient_checkpointing_enable()
    model.model.config.use_cache = False
    
    # Tracking dictionaries
    step_logs = []
    behavior_losses = []
    obf_losses = []
    loss10_behavior = []
    loss10_obf = []
    loss50_behavior = []
    loss50_obf = []
    
    step_num = 0
    total_steps = (len(label_1_data) + len(label_0_data)) * num_epochs
    pbar = tqdm(total=total_steps, desc="Enhanced Training Progress")
    
    for epoch in range(num_epochs):
        logger.print(f"Epoch {epoch}...")
        
        # Combine and shuffle data
        epoch_data = [(item, "label_1") for item in label_1_data] + [(item, "label_0") for item in label_0_data]
        random.shuffle(epoch_data)
        
        for data_item, label_type in epoch_data:
            with torch.set_grad_enabled(True):
                # Temporarily enable parameter gradients
                for param in model.parameters():
                    param.requires_grad = True
                    
                # Clear memory before forward pass
                torch.cuda.empty_cache()
                
                probe, prompt_text, response_text, token_ids, label, use_behavior_loss = data_item
                
                # Get probe name for logging
                probe_name = "unknown"
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
                    
                    # Get obfuscation loss
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
    
    return model, behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf

def train_concept_self_obfuscation_model(
    model_name: str, # e.g."gemma_2_9b_instruct"
    model_checkpoint: str, # e.g. "experiments/self_obfuscation_main/outputs/model_checkpoints/checkpoint_20250629_134811_synthetic_logistic_12",
    probe_checkpoints: str, # e.g. "20250701_002418_TRAIN_synthetic_logistic_12",
    synthetic_data_dir: str, # e.g. "data/synthetic_data/outputs/20250629_134811",
    ultrachat_json: str, # e.g. "data/self_obfuscation/outputs/ultrachat_model_training_data_20250215_023915.json",
    target_concepts: List[str], # e.g. ["excited", "german", "jokey", "all-caps", "mathematical", "chemistry-based", "biology-focused", "rhyming"]
    num_samples_per_concept: int, # e.g. 50
    num_epochs: int, # e.g. 3
    learning_rate: float, # e.g. 2e-6
    behaviour_lambda: float, # e.g. 0.1
    custom_folder_name: str = "",
    save_model_checkpoint: bool = False,
    logger = None
):
    """Train a model to perform self-obfuscation against concept probes."""
    if logger is None:
        logger = CSVTXTLogger(print_logs_to_console=True)
    
    # Default target concepts if not specified
    if target_concepts is None:
        target_concepts = ["excited", "german", "jokey", "all-caps", "mathematical", 
                          "chemistry-based", "biology-focused", "rhyming"]
    
    # Construct full paths
    probe_checkpoints_path = f"experiments/self_obfuscation_concepts/outputs/probe_checkpoints/{probe_checkpoints}"
    
    logger.print(f"Loading model: {model_checkpoint or model_name}")
    model = load_local_model(checkpoint_path=model_checkpoint, model_name=model_name)
    
    logger.print(f"Loading concept probes from: {probe_checkpoints_path}")
    concept_to_probe = load_concept_probes(probe_checkpoints_path)
    
    # Filter to only requested concepts that have probes
    available_concepts = [c for c in target_concepts if c in concept_to_probe]
    logger.print(f"Available concepts with probes: {available_concepts}")
    
    logger.print(f"Loading synthetic concept data from: {synthetic_data_dir}")
    concept_to_topical_data, concept_to_vanilla_data = load_synthetic_concept_data(synthetic_data_dir, available_concepts)
    
    logger.print(f"Loading UltraChat data from: {ultrachat_json}")
    ultrachat_data = load_ultrachat_data(ultrachat_json)
    
    # Generate training data with new N/X/Y notation
    logger.print("Generating training data with N/X/Y notation...")
    data_splits = process_concept_training_data(
        concept_to_topical_data=concept_to_topical_data,
        concept_to_vanilla_data=concept_to_vanilla_data,
        concept_to_probe=concept_to_probe,
        ultrachat_data=ultrachat_data,
        num_samples_per_concept=num_samples_per_concept,
        logger=logger
    )
    
    # Get target layer from probe metadata
    metadata_path = os.path.join(probe_checkpoints_path, "metadata.json")
    target_layers = [12]  # default
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            target_layers = metadata.get('target_layers', [12])
    
    logger.print(f"Target layers: {target_layers}")
    
    # Model training details
    model_details = {
        "model_name": model_name,
        "model_checkpoint": model_checkpoint,
        "probe_checkpoints": probe_checkpoints,
        "target_concepts": available_concepts,
        "num_samples_per_concept": num_samples_per_concept,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "behaviour_lambda": behaviour_lambda,
        "target_layers": target_layers,
        "training_data_sizes": {k: len(v) for k, v in data_splits.items()}
    }
    
    # Set up experiment directory
    output_dir_base = "experiments/self_obfuscation_concepts/outputs/model_checkpoints"
    additional_suffix = "concepts"
    
    checkpoint_path = setup_experiment_directory(
        output_dir_base=output_dir_base,
        custom_folder_name=custom_folder_name,
        experiment_type="checkpoint",
        additional_suffix=additional_suffix
    )
    
    # Prepare data for training
    label_1_data = data_splits['clue_N_data_X_probe_X_topical'] + data_splits['clue_X_data_Y_probe_Y_topical']
    label_0_data = (data_splits['clue_X_data_X_probe_X_topical'] + 
                    data_splits['clue_X_data_X_probe_X_vanilla'] + 
                    data_splits['clue_X_data_X_probe_Y_vanilla'] +
                    data_splits['clue_N_data_Y_probe_X_topical'] + 
                    data_splits['clue_N_data_Y_probe_X_vanilla'] +
                    data_splits['clue_N_data_Y_probe_X_ultrachat'] +
                    data_splits['clue_X_data_Y_probe_X_topical'] +
                    data_splits['clue_X_data_Y_probe_X_vanilla'] +
                    data_splits['clue_X_data_Y_probe_X_ultrachat'])
    
    logger.print(f"Training data prepared:")
    logger.print(f"  Label=1 (maximize probe): {len(label_1_data)}")
    logger.print(f"  Label=0 (minimize probe): {len(label_0_data)}")
    
    # Train model
    logger.print("Starting model training...")
    trained_model, behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf = concept_training_loop(
        model=model,
        label_1_data=label_1_data,
        label_0_data=label_0_data,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        behaviour_lambda=behaviour_lambda,
        target_layers=target_layers,
        logger=logger,
        checkpoint_path=checkpoint_path,
        concept_to_probe=concept_to_probe,
        data_splits=data_splits
    )
    
    # Save model checkpoint
    if save_model_checkpoint:
        logger.print(f"Saving model checkpoint to {checkpoint_path}...")
        trained_model.model.save_pretrained(checkpoint_path)
    
    # Save training metadata
    metadata = create_experiment_metadata(
        experiment_type="model_training",
        model_name=model_name,
        model_checkpoint=model_checkpoint,
        **model_details
    )
    save_metadata(checkpoint_path, metadata, filename="model_training_metadata.json")
    
    # Generate and save training plots
    logger.print("Generating training loss plots...")
    plot_path = os.path.join(checkpoint_path, "training_losses.png")
    plot_training_curves(behavior_losses, obf_losses, loss10_behavior, loss10_obf, loss50_behavior, loss50_obf, plot_path)
    
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
                      default="20250701_002418_TRAIN_synthetic_logistic_12",
                      help='Name of probe checkpoints directory')
    parser.add_argument('--synthetic_data_dir', type=str,
                      default="data/synthetic_data/outputs/20250629_134811",
                      help='Directory containing synthetic concept data')
    parser.add_argument('--ultrachat_json', type=str,
                      default="data/self_obfuscation/outputs/ultrachat_model_training_data_20250215_023915.json",
                      help='Path to UltraChat training data')
    parser.add_argument('--target_concepts', type=str, nargs='+',
                      default=["excited", "german", "jokey", "all-caps", "mathematical", 
                              "chemistry-based", "biology-focused", "rhyming"],
                      help='List of target concepts to train against')
    parser.add_argument('--num_samples_per_concept', type=int, default=50,
                      help='Number of samples per concept for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-6,
                      help='Learning rate for training')
    parser.add_argument('--behaviour_lambda', type=float, default=0.1,
                      help='Lambda parameter for behaviour loss')
    parser.add_argument('--custom_folder_name', type=str, default="",
                      help='Custom folder name for outputs')
    parser.add_argument('--save_model_checkpoint', action='store_true',
                      help='Whether to save the model checkpoint after training')
    
    args = parser.parse_args()
    
    train_concept_self_obfuscation_model(
        model_name=args.model_name,
        model_checkpoint=args.model_checkpoint,
        probe_checkpoints=args.probe_checkpoints,
        synthetic_data_dir=args.synthetic_data_dir,
        ultrachat_json=args.ultrachat_json,
        target_concepts=args.target_concepts,
        num_samples_per_concept=args.num_samples_per_concept,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        behaviour_lambda=args.behaviour_lambda,
        custom_folder_name=args.custom_folder_name,
        save_model_checkpoint=args.save_model_checkpoint
    )