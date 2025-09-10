#!/usr/bin/env python3
"""
Script to create violin plots of text length by adjective and cosine similarity bar plots.
Reads evaluated synthetic data and plots text length distribution for each adjective
where texts have a normalized rating >= threshold.
Also includes UltraChat comparison and cosine similarity analysis.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import gc

def load_and_process_data(filepath, text_field, rating_threshold=1.0):
    """Load JSON data and process for violin plot."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Collect text lengths for each adjective
    adjective_data = defaultdict(list)
    
    for item in data:
        text = item[text_field]
        text_length = len(text)
        ratings = item['prompt_normalized_ratings']
        
        # Debug: check for negative or zero lengths
        if text_length <= 0:
            print(f"Warning: {text_field} length is {text_length} for text: '{text[:50]}...'")
            continue
        
        # Add text length to adjectives with rating >= threshold
        for adjective, rating in ratings.items():
            if rating is not None and rating >= rating_threshold:
                adjective_data[adjective].append(text_length)
    
    # Filter out adjectives with too few data points for meaningful violin plots
    min_samples = 5
    filtered_data = {adj: lengths for adj, lengths in adjective_data.items() 
                    if len(lengths) >= min_samples}
    
    # Debug: check for any negative values in the filtered data
    all_lengths = []
    for adj, lengths in filtered_data.items():
        min_len = min(lengths)
        max_len = max(lengths)
        all_lengths.extend(lengths)
        if min_len < 0:
            print(f"Warning: {adj} has negative lengths: min={min_len}, max={max_len}")
    
    if all_lengths:
        overall_min = min(all_lengths)
        overall_max = max(all_lengths)
        print(f"Overall range: {overall_min} to {overall_max} characters")
        if overall_min < 0:
            print(f"ERROR: Found negative {text_field} lengths! Min: {overall_min}")
    
    return filtered_data

def load_ultrachat_sample(num_conversations=10000, split="test_gen"):
    """Load a sample from UltraChat dataset."""
    print(f"Loading {num_conversations} UltraChat conversations...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, trust_remote_code=True)
    sampled_dataset = dataset.select(range(min(num_conversations, len(dataset))))
    return sampled_dataset

def extract_prompt_response_from_conversation(conversation, cut_prompt_to_first_and_last_sentence=False, cut_response_to_first_sentence=False, minimum_response_cut_length=-1):
    """Extract a prompt/response pair from a conversation, with safety checks."""
    if not conversation or len(conversation) < 2: 
        return None, None
    prompt_msg, response_msg = conversation[0], conversation[1]
    if not isinstance(prompt_msg, dict) or not isinstance(response_msg, dict): 
        return None, None
    prompt, response = prompt_msg.get("content"), response_msg.get("content")
    if not prompt or not response: 
        return None, None
    if cut_prompt_to_first_and_last_sentence: 
        prompt = cut_to_first_and_last_sentence(prompt)
    if cut_response_to_first_sentence: 
        response = cut_to_first_sentence(response, minimum_response_cut_length)
    return prompt, response

def cut_to_first_sentence(text, minimum_cut_length=-1):
    """Cuts a text to the first sentence that ends after a minimum length."""
    end_chars = '.!?\n'
    start_search = minimum_cut_length if minimum_cut_length > 0 else 0
    first_end_index = -1
    for char in end_chars:
        index = text.find(char, start_search)
        if index != -1 and (first_end_index == -1 or index < first_end_index):
            first_end_index = index
    if first_end_index != -1: 
        return text[:first_end_index+1].strip()
    if minimum_cut_length > 0:
        first_end_index_any = -1
        for char in end_chars:
            index = text.find(char)
            if index != -1 and (first_end_index_any == -1 or index < first_end_index_any):
                first_end_index_any = index
        if first_end_index_any != -1: 
            return text[:first_end_index_any+1].strip()
    return text.strip()

def cut_to_first_and_last_sentence(text):
    """Cut a text to just the first and last sentences."""
    sentences = []
    start = 0
    for i, char in enumerate(text):
        if char in '.!?\n':
            sentences.append(text[start:i+1].strip())
            start = i+1
    if start < len(text):
        ending = text[start:].strip()
        if ending: 
            sentences.append(ending)
    if not sentences: 
        return text
    elif len(sentences) == 1: 
        return sentences[0]
    else: 
        return sentences[0] + " " + sentences[-1]

def get_ultrachat_data(text_field):
    """Get UltraChat data for the specified text field."""
    ultrachat_dataset = load_ultrachat_sample(10000)
    ultrachat_texts = []
    
    for item in tqdm(ultrachat_dataset, desc="Processing UltraChat conversations"):
        conversation = item['messages']
        prompt, response = extract_prompt_response_from_conversation(
            conversation, cut_prompt_to_first_and_last_sentence=True,
            cut_response_to_first_sentence=True, minimum_response_cut_length=100
        )
        
        if prompt and response:
            if text_field == 'prompt':
                ultrachat_texts.append(prompt)
            elif text_field in ['vanilla_response', 'topical_response']:
                ultrachat_texts.append(response)
    
    print(f"Collected {len(ultrachat_texts)} UltraChat {text_field} texts")
    return ultrachat_texts

def calculate_cosine_similarity_metrics(texts, model_name='all-MiniLM-L6-v2', batch_size=64):
    """Calculate average pairwise cosine similarity for a list of texts."""
    if len(texts) < 2:
        return None
    
    print(f"Calculating cosine similarity for {len(texts)} texts...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    
    # Encode texts
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    
    # Calculate pairwise cosine similarity
    cos_sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    # Get the average of the upper triangle (excluding the diagonal)
    upper_triangle_indices = np.triu_indices_from(cos_sim_matrix, k=1)
    avg_pairwise_sim = np.mean(cos_sim_matrix[upper_triangle_indices])
    
    # Clean up
    del embeddings, cos_sim_matrix
    gc.collect()
    torch.cuda.empty_cache()
    
    return float(avg_pairwise_sim)

def create_violin_plot(adjective_data, ultrachat_data, output_path, text_type, rating_threshold):
    """Create violin plot of text lengths by adjective with UltraChat comparison."""
    # Prepare data for seaborn
    plot_data = []
    for adjective, lengths in adjective_data.items():
        for length in lengths:
            plot_data.append({'adjective': adjective, 'text_length': length})
    
    # Add UltraChat data
    if ultrachat_data:
        for length in ultrachat_data:
            plot_data.append({'adjective': 'UltraChat', 'text_length': length})
    
    df = pd.DataFrame(plot_data)
    
    # Sort adjectives by median text length for better visualization
    median_lengths = df.groupby('adjective')['text_length'].median().sort_values()
    sorted_adjectives = median_lengths.index.tolist()
    
    # Create the plot
    plt.figure(figsize=(16, 8))
    
    # Create violin plot with different colors for UltraChat
    palette = ['red' if adj == 'UltraChat' else 'lightblue' for adj in sorted_adjectives]
    ax = sns.violinplot(data=df, x='adjective', y='text_length', 
                       order=sorted_adjectives, inner='quartile', palette=palette)
    
    # Customize the plot
    plt.title(f'Distribution of {text_type.title()} Length by Adjective\n(Normalized Rating ≥ {rating_threshold}, UltraChat in Red)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Adjective', fontsize=12, fontweight='bold')
    plt.ylabel(f'{text_type.title()} Length (characters)', fontsize=12, fontweight='bold')
    
    # Set y-axis to start at 0 to avoid negative display issues
    plt.ylim(bottom=0)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # Display statistics
    print(f"\nStatistics for {text_type} (threshold {rating_threshold}):")
    print(f"Total adjectives plotted: {len(adjective_data)}")
    for adj in sorted_adjectives[:10]:  # Show top 10 for brevity
        count = len([x for x in plot_data if x['adjective'] == adj])
        median_len = median_lengths[adj]
        print(f"  {adj}: {count} texts, median length: {median_len:.0f} chars")
    if len(sorted_adjectives) > 10:
        print(f"  ... and {len(sorted_adjectives) - 10} more adjectives")
    
    plt.close()  # Close the plot to free memory

def create_cosine_similarity_bar_plot(filepath, text_field, rating_threshold, plots_dir):
    """Create bar plot of cosine similarity for each adjective."""
    print(f"\nCreating cosine similarity bar plot for {text_field} (threshold {rating_threshold})")
    
    # Load and process data for cosine similarity
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    adjective_texts = defaultdict(list)
    
    for item in data:
        text = item[text_field]
        ratings = item['prompt_normalized_ratings']
        
        if len(text) <= 0:
            continue
        
        # Add text to adjectives with rating >= threshold
        for adjective, rating in ratings.items():
            if rating is not None and rating >= rating_threshold:
                adjective_texts[adjective].append(text)
    
    # Filter adjectives with enough samples and limit to top adjectives by count
    min_samples = 20  # Need more samples for cosine similarity
    filtered_adjectives = {adj: texts for adj, texts in adjective_texts.items() 
                          if len(texts) >= min_samples}
    
    # Limit to top 20 adjectives by sample count for computational efficiency
    sorted_by_count = sorted(filtered_adjectives.items(), key=lambda x: len(x[1]), reverse=True)
    filtered_adjectives = dict(sorted_by_count[:20])
    
    if not filtered_adjectives:
        print(f"No adjectives found with >= {min_samples} samples for cosine similarity calculation")
        return
    
    print(f"Calculating cosine similarity for top {len(filtered_adjectives)} adjectives by sample count...")
    
    # Calculate cosine similarity for each adjective
    similarity_results = {}
    for adjective, texts in tqdm(filtered_adjectives.items(), desc="Processing adjectives"):
        # Sample down to max 200 texts for computational efficiency
        if len(texts) > 200:
            texts = np.random.choice(texts, 200, replace=False).tolist()
        
        similarity = calculate_cosine_similarity_metrics(texts, batch_size=16)
        if similarity is not None:
            similarity_results[adjective] = similarity
    
    # Get UltraChat comparison
    print("Getting UltraChat comparison data for cosine similarity...")
    ultrachat_texts = get_ultrachat_data(text_field)
    if len(ultrachat_texts) > 200:
        ultrachat_texts = np.random.choice(ultrachat_texts, 200, replace=False).tolist()
    
    ultrachat_similarity = calculate_cosine_similarity_metrics(ultrachat_texts, batch_size=16)
    if ultrachat_similarity is not None:
        similarity_results['UltraChat'] = ultrachat_similarity
    
    # Create bar plot
    adjectives = list(similarity_results.keys())
    similarities = list(similarity_results.values())
    
    plt.figure(figsize=(15, 8))
    
    # Create bar plot with UltraChat highlighted
    colors = ['red' if adj == 'UltraChat' else 'lightblue' for adj in adjectives]
    bars = plt.bar(range(len(adjectives)), similarities, color=colors)
    
    # Customize plot
    plt.title(f'Average Pairwise Cosine Similarity by Adjective\n{text_field.replace("_", " ").title()} (Threshold ≥ {rating_threshold}, UltraChat in Red)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Adjective', fontsize=12, fontweight='bold')
    plt.ylabel('Average Pairwise Cosine Similarity', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(adjectives)), adjectives, rotation=45, ha='right')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{sim:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    safe_text_name = text_field.replace('_', '-')
    output_filename = f"{safe_text_name}_cosine_similarity_threshold_{rating_threshold}.png"
    output_path = os.path.join(plots_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cosine similarity plot saved as {output_path}")
    
    # Print results
    print(f"\nCosine Similarity Results for {text_field} (threshold {rating_threshold}):")
    sorted_results = sorted(similarity_results.items(), key=lambda x: x[1])
    for adj, sim in sorted_results:
        print(f"  {adj}: {sim:.4f}")
    
    plt.close()

def main():
    import sys
    
    # Check if user wants to skip cosine similarity calculations
    skip_cosine = len(sys.argv) > 1 and sys.argv[1] == '--skip-cosine'
    
    filepath = "data/self_obfuscation_v1_synthetic_rating/outputs/20250629_134811/evaluated/mega_dataset_evaluated_20250716_190901.json"
    
    # Create plots directory
    plots_dir = "data/self_obfuscation_v1_synthetic_rating/outputs/20250629_134811/evaluated/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define text fields and thresholds to process
    text_fields = [
        ('prompt', 'Prompt'),
        ('vanilla_response', 'Vanilla Response'),
        ('topical_response', 'Topical Response')
    ]
    
    thresholds = [0.75, 1.0]
    
    # Cache UltraChat data to avoid reloading
    ultrachat_cache = {}
    
    for text_field, text_name in text_fields:
        # Load UltraChat data once per text field
        if text_field not in ultrachat_cache:
            print(f"\n{'='*60}")
            print(f"Loading UltraChat data for {text_name}")
            print(f"{'='*60}")
            ultrachat_cache[text_field] = get_ultrachat_data(text_field)
        
        for threshold in thresholds:
            print(f"\n{'='*60}")
            print(f"Processing {text_name} with threshold {threshold}")
            print(f"{'='*60}")
            
            # Load and process data
            adjective_data = load_and_process_data(filepath, text_field, threshold)
            
            if not adjective_data:
                print(f"No adjectives found with rating >= {threshold} and sufficient samples for {text_name}.")
                continue
            
            print(f"Found {len(adjective_data)} adjectives with sufficient data points.")
            
            # Create violin plot with UltraChat comparison
            safe_text_name = text_name.lower().replace(' ', '_')
            output_filename = f"{safe_text_name}_length_violin_plot_threshold_{threshold}.png"
            output_path = os.path.join(plots_dir, output_filename)
            
            ultrachat_lengths = [len(text) for text in ultrachat_cache[text_field]]
            create_violin_plot(adjective_data, ultrachat_lengths, output_path, text_name, threshold)
            
            # Create cosine similarity bar plot (unless skipped)
            if not skip_cosine:
                create_cosine_similarity_bar_plot(filepath, text_field, threshold, plots_dir)
            else:
                print("Skipping cosine similarity calculation (--skip-cosine flag provided)")

if __name__ == "__main__":
    print("Usage: python script.py [--skip-cosine]")
    print("--skip-cosine: Skip the computationally expensive cosine similarity calculations")
    main()