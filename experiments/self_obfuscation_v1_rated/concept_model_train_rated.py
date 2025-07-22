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
import json
import torch
import argparse
import os
import pickle
import sys
from typing import List, Dict
import glob
from pathlib import Path
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
from data.data import PosNegDataByCategory, PromptRespRating
from experiments.self_obfuscation_v1.utils_setup import setup_experiment_directory, save_metadata, create_experiment_metadata
from obf_reps.models.hf import HFHardPromptedWithSelectableLayers

# Concept synonyms dictionary - each concept maps to a list of synonyms including the original
CONCEPT_SYNONYMS = {
    "HTML": ["HTML", "hypertext markup", "web markup", "HTML format", "markup language"],
    "abstract": ["abstract", "theoretical", "conceptual", "intangible", "philosophical"],
    "academic": ["academic", "scholarly", "educational", "formal", "university-style"],
    "all-caps": ["all-caps", "uppercase", "capitalized", "shouting", "bold capitals"],
    "angry": ["angry", "furious", "mad", "irritated", "hostile"],
    "ascii-art": ["ascii-art", "text art", "character art", "symbol graphics", "text graphics"],
    "biology-focused": ["biology-focused", "biological", "life sciences", "biologically-themed", "bio-centric"],
    "boring": ["boring", "dull", "tedious", "uninteresting", "mundane"],
    "capitalised": ["capitalised", "capitalized", "title-cased", "proper-cased", "initial-caps"],
    "casual": ["casual", "informal", "relaxed", "conversational", "laid-back"],
    "chemical": ["chemical", "chemistry-related", "molecular", "compound-based", "chemical-themed"],
    "chemistry-based": ["chemistry-based", "chemical", "molecular", "compound-focused", "chemistry-themed"],
    "chemistry-focused": ["chemistry-focused", "chemical", "chemistry-based", "molecular", "compound-oriented"],
    "chinese": ["chinese", "mandarin", "cantonese", "sino", "chinese-language"],
    "comforting": ["comforting", "soothing", "reassuring", "calming", "supportive"],
    "commanding": ["commanding", "authoritative", "directive", "imperative", "controlling"],
    "confused": ["confused", "bewildered", "puzzled", "perplexed", "disoriented"],
    "customer-service-roleplay": ["customer-service-roleplay", "support roleplay", "service simulation", "helpdesk roleplay", "customer-support"],
    "czech": ["czech", "bohemian", "moravian", "czech-language", "czechian"],
    "dutch": ["dutch", "netherlands", "flemish", "dutch-language", "nederlander"],
    "excited": ["excited", "enthusiastic", "thrilled", "energetic", "animated"],
    "finnish": ["finnish", "suomi", "finn", "finnish-language", "scandinavian-finnish"],
    "formal": ["formal", "official", "professional", "structured", "ceremonial"],
    "french": ["french", "francophone", "gallic", "french-style", "franco"],
    "french-language": ["french-language", "francophone", "french", "en francais", "gallic-language"],
    "french-language-only": ["french-language-only", "purely french", "exclusively francophone", "only french", "french-exclusive"],
    "fun": ["fun", "playful", "entertaining", "amusing", "enjoyable"],
    "german": ["german", "deutsch", "germanic", "teutonic", "german-language"],
    "german-language": ["german-language", "deutsch", "german", "germanic-language", "teutonic-language"],
    "german-language-only": ["german-language-only", "purely german", "exclusively deutsch", "only german", "german-exclusive"],
    "gibberish": ["gibberish", "nonsense", "babble", "jargon", "meaningless"],
    "greek": ["greek", "hellenic", "grecian", "greek-language", "attic"],
    "harmful": ["harmful", "dangerous", "malicious", "destructive", "toxic"],
    "helpful": ["helpful", "useful", "beneficial", "supportive", "constructive"],
    "humorous": ["humorous", "funny", "witty", "comedic", "amusing"],
    "imperative": ["imperative", "commanding", "directive", "mandatory", "instructional"],
    "japanese": ["japanese", "nihongo", "nippon", "japanese-language", "yamato"],
    "javascript": ["javascript", "JS", "ECMAScript", "node", "web scripting"],
    "jokey": ["jokey", "humorous", "playful", "comedic", "lighthearted"],
    "json-formatted": ["json-formatted", "JSON", "structured data", "javascript object", "data format"],
    "korean": ["korean", "hangul", "korean-language", "hanguk", "seoul-style"],
    "lazy": ["lazy", "sluggish", "lethargic", "indolent", "unmotivated"],
    "literature-focused": ["literature-focused", "literary", "textual", "book-oriented", "prose-based"],
    "markdown-formatted": ["markdown-formatted", "markdown", "MD format", "markup text", "structured text"],
    "mathematical": ["mathematical", "numeric", "quantitative", "algebraic", "computational"],
    "personality-analysis": ["personality-analysis", "psychological profiling", "character assessment", "personality evaluation", "trait analysis"],
    "philosophical": ["philosophical", "contemplative", "metaphysical", "theoretical", "existential"],
    "poetic": ["poetic", "lyrical", "verse-like", "rhythmic", "artistic"],
    "polish": ["polish", "polski", "polish-language", "slavic-polish", "warsaw-style"],
    "python": ["python", "pythonic", "snake-language", "py", "python-code"],
    "question-asking": ["question-asking", "interrogative", "inquiry-based", "questioning", "inquisitive"],
    "question-laden": ["question-laden", "question-heavy", "inquiry-filled", "interrogative-dense", "question-rich"],
    "questioning": ["questioning", "inquisitive", "interrogative", "inquiry-based", "doubt-filled"],
    "reassuring": ["reassuring", "comforting", "calming", "supportive", "encouraging"],
    "reversed": ["reversed", "backwards", "inverted", "flipped", "mirror-text"],
    "rhyming": ["rhyming", "poetic", "verse", "rhythmic", "melodic"],
    "romanian": ["romanian", "daco-romanian", "romanian-language", "moldovan", "vlach"],
    "short-sentence-only": ["short-sentence-only", "brief sentences", "concise statements", "terse responses", "minimal sentences"],
    "sleepy": ["sleepy", "drowsy", "tired", "lethargic", "weary"],
    "slovak": ["slovak", "slovakian", "slovak-language", "slavic-slovak", "bratislava-style"],
    "spanish": ["spanish", "espanol", "castilian", "hispanic", "iberian"],
    "supportive": ["supportive", "encouraging", "helpful", "caring", "understanding"],
    "therapeutic": ["therapeutic", "healing", "counseling", "psychological", "restorative"],
    "title-case": ["title-case", "proper case", "headline style", "capitalized words", "title format"]
}

def get_random_synonym(concept: str) -> str:
    """Get a random synonym for the given concept, falling back to the original if not found."""
    synonyms = CONCEPT_SYNONYMS.get(concept, [concept])
    return random.choice(synonyms)

def plot_validation_curves(validation_logs: List[Dict], save_path: str):
    """Generate and save a dedicated validation loss plot."""
    if not validation_logs:
        return
    
    val_steps = [log['step'] for log in validation_logs]
    val_beh_losses = [log['avg_beh_loss'] for log in validation_logs]
    val_obf_losses = [log['avg_obf_loss'] for log in validation_logs]
    
    # Create 2x1 subplot layout for validation losses
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Validation behavior loss
    ax1.plot(val_steps, val_beh_losses, 'o-', label='Validation Behavior Loss', 
             color='lightblue', markersize=4, linewidth=2)
    ax1.set_title('Validation Behavior Loss Over Training')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Behavior Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation obfuscation loss
    ax2.plot(val_steps, val_obf_losses, 'o-', label='Validation Obfuscation Loss', 
             color='orange', markersize=4, linewidth=2)
    ax2.set_title('Validation Obfuscation Loss Over Training')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Obfuscation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined validation plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if val_beh_losses:
        ax.plot(val_steps, val_beh_losses, 'o-', label='Validation Behavior Loss', 
               color='lightblue', markersize=4, linewidth=2)
    
    if val_obf_losses:
        ax.plot(val_steps, val_obf_losses, 'o-', label='Validation Obfuscation Loss', 
               color='orange', markersize=4, linewidth=2)
    
    ax.set_title('Validation Loss Curves')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    combined_val_path = save_path.replace('.png', '_combined.png')
    plt.savefig(combined_val_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plots saved to {save_path} and {combined_val_path}")

def load_concept_probes(probe_checkpoints_path: str) -> Dict[str, object]:
    """Load trained concept probes from checkpoint directory."""
    print(f"Loading concept probes from: {probe_checkpoints_path}")
    concept_to_probe = {}
    probe_files = glob.glob(os.path.join(probe_checkpoints_path, "*.pkl"))
    print(f"Found {len(probe_files)} probe files")
    
    # Create module mapping to handle renamed modules
    original_modules = sys.modules.copy()
    try:
        # Map old module names to new ones
        print("Setting up module mapping for backwards compatibility...")
        if 'experiments.self_obfuscation_main' not in sys.modules:
            print("Mapping experiments.self_obfuscation_main to experiments.self_obfuscation_v0")
            sys.modules['experiments.self_obfuscation_main'] = sys.modules.get('experiments.self_obfuscation_v0', None)
            if sys.modules['experiments.self_obfuscation_main'] is None:
                print("Importing experiments.self_obfuscation_v0")
                try:
                    import experiments.self_obfuscation_v0
                    sys.modules['experiments.self_obfuscation_main'] = experiments.self_obfuscation_v0
                except ImportError as e:
                    print(f"Error importing experiments.self_obfuscation_v0: {e}")
                    raise
        
        for probe_file in probe_files:
            concept = Path(probe_file).stem
            print(f"Loading probe for concept: {concept}")
            try:
                with open(probe_file, 'rb') as f:
                    probe = pickle.load(f)
                concept_to_probe[concept] = probe
                print(f"Successfully loaded probe for {concept}")
            except (IOError, pickle.PickleError) as e:
                print(f"Error loading probe file {probe_file}: {e}")
                continue
    except Exception as e:
        print(f"Unexpected error loading probes: {e}")
        raise
    finally:
        # Clean up temporary module mapping
        print("Cleaning up temporary module mapping...")
        if 'experiments.self_obfuscation_main' in sys.modules and 'experiments.self_obfuscation_main' not in original_modules:
            del sys.modules['experiments.self_obfuscation_main']
    
    print(f"Finished loading {len(concept_to_probe)} concept probes")
    return concept_to_probe

def process_concept_training_data(
    concept_to_topical_data: PosNegDataByCategory,
    concept_to_vanilla_data: PosNegDataByCategory,
    concept_to_ultrachat_data: PosNegDataByCategory,
    concept_to_probe: Dict[str, object],
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
    
    concepts = list(concept_to_probe.keys())
    
    # Create intersection sets for concept selection to avoid KeyError
    concepts_with_topical = list(set(concept_to_probe.keys()) & set(concept_to_topical_data.categories.keys()))
    concepts_with_vanilla = list(set(concept_to_probe.keys()) & set(concept_to_vanilla_data.categories.keys()))
    
    if logger:
        logger.print(f"Concepts with probes: {len(concepts)}")
        logger.print(f"Concepts with topical data: {len(concepts_with_topical)}")
        logger.print(f"Concepts with vanilla data: {len(concepts_with_vanilla)}")
        
        if len(concepts_with_topical) == 0:
            logger.print("WARNING: No concepts have both probes and topical data!")
        if len(concepts_with_vanilla) == 0:
            logger.print("WARNING: No concepts have both probes and vanilla data!")
    
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
                    split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, False))
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
                    clued_prompt = f"You are being probed for '{get_random_synonym(concept_x)}'. {prompt_resp.prompt}"
                    split_samples.append((probe_y, clued_prompt, prompt_resp.response, None, 1, False))
                    
            elif split_name == 'clue_X_data_X_probe_X_topical':
                # 3. X clue, X topical data, X probe → target score 0
                if not concepts_with_topical:
                    if logger and attempts % 100 == 0:
                        logger.print(f"  Failed attempt {attempts} for {split_name}: no concepts with topical data available")
                    continue
                concept_x = random.choice(concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_x = concept_to_topical_data.categories[concept_x]
                prompt_resp = random.choice(topical_data_x.pos_dataset)
                clued_prompt = f"You are being probed for '{get_random_synonym(concept_x)}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, False))
                    
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
                clued_prompt = f"You are being probed for '{get_random_synonym(concept_x)}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_X_data_X_probe_Y_vanilla':
                # 5. X clue, X vanilla data, Y probe → target score 0 (+ beh loss)
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
                clued_prompt = f"You are being probed for '{get_random_synonym(concept_x)}'. {prompt_resp.prompt}"
                split_samples.append((probe_y, clued_prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_N_data_Y_probe_X_topical':
                # 6. No clue, Y topical data, X probe → target score 0
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
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 0, False))
                    
            elif split_name == 'clue_N_data_Y_probe_X_vanilla':
                # 7. No clue, Y vanilla data, X probe → target score 0 (+ beh loss)
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
                # 8. No clue, ultrachat data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                probe_x = concept_to_probe[concept_x]
                if all_ultrachat_data:
                    prompt_resp = random.choice(all_ultrachat_data)
                    split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_X_data_Y_probe_X_topical':
                # 9. X clue, Y topical data, X probe → target score 0
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
                clued_prompt = f"You are being probed for '{get_random_synonym(concept_x)}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, False))
                    
            elif split_name == 'clue_X_data_Y_probe_X_vanilla':
                # 10. X clue, Y vanilla data, X probe → target score 0 (+ beh loss)
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
                clued_prompt = f"You are being probed for '{get_random_synonym(concept_x)}'. {prompt_resp.prompt}"
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))
                    
            elif split_name == 'clue_X_data_Y_probe_X_ultrachat':
                # 11. X clue, ultrachat data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                probe_x = concept_to_probe[concept_x]
                if all_ultrachat_data:
                    prompt_resp = random.choice(all_ultrachat_data)
                    clued_prompt = f"You are being probed for '{get_random_synonym(concept_x)}'. {prompt_resp.prompt}"
                    split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))
        
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

def evaluate_validation_set(
    model: HFHardPromptedWithSelectableLayers,
    val_data: List,
    target_layers: List[int],
    behaviour_lambda: float,
    step_num: int,
    logger
) -> Dict:
    """Evaluate model on validation set and return loss metrics."""
    model.eval()
    total_obf_loss = 0.0
    total_beh_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data_item in val_data:
            probe, prompt_text, response_text, token_ids, label, use_behavior_loss = data_item
            
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
            
            # Calculate behavior loss
            if use_behavior_loss:
                beh_losses = F.cross_entropy(
                    behavior_output.target_logits.transpose(1, 2),
                    behavior_target_input_ids,
                    reduction="none"
                )
                beh_losses *= behavior_output.loss_mask.float()
                num_valid_tokens = behavior_output.loss_mask.float().sum(dim=-1)
                behavior_loss = beh_losses.sum(dim=-1) / (num_valid_tokens + 1e-8)
            else:
                behavior_loss = torch.tensor(0.0, device=behavior_output.target_logits.device)
            
            # Calculate obfuscation loss
            scores = probe.predict(reps=behavior_output.target_reps)
            ideal_scores = torch.tensor([label], device=model.device, dtype=torch.float16)
            obf_loss = F.mse_loss(scores, ideal_scores, reduction="none")
            
            total_obf_loss += obf_loss.mean().item()
            total_beh_loss += behavior_loss.mean().item() if hasattr(behavior_loss, 'mean') else behavior_loss.item()
            total_samples += 1
    
    model.train()
    
    validation_result = {
        'step': step_num,
        'avg_obf_loss': total_obf_loss / total_samples,
        'avg_beh_loss': total_beh_loss / total_samples,
        'num_samples': total_samples
    }
    
    logger.print(f"Validation at step {step_num}: obf_loss={validation_result['avg_obf_loss']:.4f}, beh_loss={validation_result['avg_beh_loss']:.4f}")
    
    return validation_result

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
    data_splits: Dict = None
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
            model, val_data, target_layers, behaviour_lambda, step_num, logger
        )
        validation_logs.append(validation_result)

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
                
                # Perform validation at specified frequency
                if val_data and step_num > 0 and step_num % validation_frequency == 0:
                    validation_result = evaluate_validation_set(
                        model, val_data, target_layers, behaviour_lambda, step_num, logger
                    )
                    validation_logs.append(validation_result)
                
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
    max_neg_rating: float = 0.25, # Maximum rating threshold for negative samples
    min_pos_rating: float = 0.75, # Minimum rating threshold for positive samples
    validation_split: float = 0.05, # Percentage of training data to hold out for validation
    validation_frequency: int = 1000, # Validate every N steps
    custom_folder_name: str = "",
    filter_to_concepts: List[str] = None,
    logger = None,
    save_model_checkpoint: bool = True,
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
    if filter_to_concepts and len(filter_to_concepts) > 0:
        concept_to_probe = {k: v for k, v in concept_to_probe.items() if k in filter_to_concepts}
        logger.print(f"Filtered to concepts: {list(concept_to_probe.keys())}")
    
    
    logger.print("Loading synthetic concept data...")
    # Load using the new data loading approach
    topical_ratings = load_rated_data(
        dataset_name=r"mega_dataset_evaluated_.*\.json$",
        response_type="topical",
        manual_path_confirm=False,
        exclude_refusals=True,
        exclude_missing_ratings=True
    )
    vanilla_ratings = load_rated_data(
        dataset_name=r"mega_dataset_evaluated_.*\.json$",
        response_type="vanilla",
        manual_path_confirm=False,
        exclude_refusals=True,
        exclude_missing_ratings=True
    )
    ultrachat_ratings = load_rated_data(
        dataset_name=r"ultrachat_mega_dataset_.*\.json$",
        response_type="vanilla",
        manual_path_confirm=False,
        exclude_refusals=True,
        exclude_missing_ratings=True
    )
    
    # Convert to the format expected by process_concept_training_data
    concept_to_topical_data = PosNegDataByCategory.from_ratings(
        topical_ratings, 
        max_neg_rating=max_neg_rating, 
        min_pos_rating=min_pos_rating
    )
    concept_to_vanilla_data = PosNegDataByCategory.from_ratings(
        vanilla_ratings, 
        max_neg_rating=max_neg_rating, 
        min_pos_rating=min_pos_rating
    )
    concept_to_ultrachat_data = PosNegDataByCategory.from_ratings(
        ultrachat_ratings, 
        max_neg_rating=max_neg_rating, 
        min_pos_rating=min_pos_rating
    )

    if filter_to_concepts and len(filter_to_concepts) > 0:
        concept_to_topical_data.categories = {k: v for k, v in concept_to_topical_data.categories.items() if k in filter_to_concepts}
        concept_to_vanilla_data.categories = {k: v for k, v in concept_to_vanilla_data.categories.items() if k in filter_to_concepts}
        concept_to_ultrachat_data.categories = {k: v for k, v in concept_to_ultrachat_data.categories.items() if k in filter_to_concepts}
    
    topical_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_topical_data.categories.items()}
    vanilla_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_vanilla_data.categories.items()}
    ultrachat_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_ultrachat_data.categories.items()}
    logger.print(f"Raw counts per category for topical data: {topical_counts}")
    logger.print(f"Raw counts per category for vanilla data: {vanilla_counts}")
    logger.print(f"Raw counts per category for ultrachat data: {ultrachat_counts}")

    min_samples = 10
    logger.print(f"Culling categories with fewer than {min_samples} samples...")
    # If any category has pos or neg dataset with fewer than 10 samples, remove it
    concept_to_topical_data.categories = {k: v for k, v in concept_to_topical_data.categories.items() if len(v.pos_dataset) > 10 and len(v.neg_dataset) > 10}
    concept_to_vanilla_data.categories = {k: v for k, v in concept_to_vanilla_data.categories.items() if len(v.pos_dataset) > 10 and len(v.neg_dataset) > 10}
    concept_to_ultrachat_data.categories = {k: v for k, v in concept_to_ultrachat_data.categories.items() if len(v.pos_dataset) > 10 and len(v.neg_dataset) > 10}

    # Generate training data with new N/X/Y notation
    logger.print("Generating training data with N/X/Y notation...")
    data_splits = process_concept_training_data(
        concept_to_topical_data=concept_to_topical_data,
        concept_to_vanilla_data=concept_to_vanilla_data,
        concept_to_ultrachat_data=concept_to_ultrachat_data,
        concept_to_probe=concept_to_probe,
        num_samples_per_concept=num_samples_per_concept,
        logger=logger
    )
    
    # Get target layer from probe metadata
    metadata_path = os.path.join(probe_checkpoints, "metadata.json")
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
        "num_samples_per_concept": num_samples_per_concept,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "behaviour_lambda": behaviour_lambda,
        "max_neg_rating": max_neg_rating,
        "min_pos_rating": min_pos_rating,
        "target_layers": target_layers,
        "training_data_sizes": {k: len(v) for k, v in data_splits.items()}
    }
    
    # Set up experiment directory
    output_dir_base = os.path.join(os.path.dirname(__file__), "outputs", "model_checkpoints")
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
        data_splits=data_splits
    )
    
    # Save model checkpoint
    if save_model_checkpoint:
        logger.print(f"Saving model checkpoint to {checkpoint_path}...")
        trained_model.model.save_pretrained(checkpoint_path)
    else:
        logger.print("Not saving model checkpoint")
    
    # Save training metadata
    metadata = create_experiment_metadata(
        experiment_type="model_training",
        **model_details
    )
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
                      default="experiments/self_obfuscation_v1/outputs/probe_checkpoints/20250717_012848_TRAIN_labelled_topical_logistic_12_075025",
                      help='Name of probe checkpoints directory')
    parser.add_argument('--num_samples_per_concept', type=int, default=150,
                      help='Number of samples per concept for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-6,
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
    
    args = parser.parse_args()
    
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
    )