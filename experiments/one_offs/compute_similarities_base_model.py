#!/usr/bin/env python3
"""
Compute cosine similarities using base Gemma model and regenerate plot
Since checkpoint model files are missing, we'll use base model
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

# Set style
sns.set_context("paper")
sns.set_style("darkgrid")
plt.rcParams['figure.constrained_layout.use'] = True

def is_deception_related(condition):
    """Check if a condition contains deception-related words"""
    deception_words = ['lie', 'lies', 'lying', 'deception', 'deceive', 'deceiving', 
                       'mislead', 'misleading', 'dishonest', 'dishonesty', 'false', 
                       'falsify', 'fake', 'fraud', 'cheat', 'cheating', 'deceptive', 'untruthful']
    return any(word in condition.lower() for word in deception_words)

def is_truth_related(condition):
    """Check if a condition contains truth-related words"""
    truth_words = ['truthful', 'honesty-focused', 'truth-focused', 'honest', 'truth']
    return any(word in condition.lower() for word in truth_words)

def get_base_gemma_embeddings(words):
    """Get embeddings using base Gemma model"""
    model_path = "/workspace/gemma_2_9b_instruct"
    print(f"Loading base Gemma model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    embeddings = {}
    
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
            summed = torch.sum(masked_embeddings, dim=1)
            lengths = torch.sum(attention_mask, dim=1, keepdim=True)
            embedding = summed / lengths
            
            embeddings[word] = embedding.cpu().numpy().flatten()
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return embeddings

def compute_similarities(embeddings, reference_word):
    """Compute cosine similarities"""
    reference_embedding = embeddings[reference_word].reshape(1, -1)
    similarities = {}
    
    for word, embedding in embeddings.items():
        if word != reference_word:
            word_embedding = embedding.reshape(1, -1)
            similarity = cosine_similarity(reference_embedding, word_embedding)[0][0]
            similarities[word] = similarity
    
    return similarities

def plot_and_save(condition_data, similarities, reference_condition, output_dir, model_note="Base Gemma"):
    """Create and save the plot"""
    
    # Prepare data
    conditions = []
    recall_scores = []
    similarity_scores = []
    colors = []
    
    for condition, data in condition_data.items():
        if condition != reference_condition and condition in similarities:
            conditions.append(condition)
            recall_scores.append(data['recall_mean'])
            similarity_scores.append(similarities[condition])
            
            if is_deception_related(condition):
                colors.append('red')
            elif is_truth_related(condition):
                colors.append('green')
            else:
                colors.append('steelblue')
    
    # Create plot
    plt.figure(figsize=(6.75, 4.5))
    
    plt.scatter(similarity_scores, recall_scores, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add reference point
    ref_recall = condition_data[reference_condition]['recall_mean']
    ref_color = 'red' if is_deception_related(reference_condition) else 'orange'
    plt.scatter(1.0, ref_recall, c=ref_color, s=150, marker='*', 
               edgecolors='darkred', linewidth=2, 
               label=f'{reference_condition} (reference)')
    
    # Add labels
    for i, condition in enumerate(conditions):
        plt.annotate(condition, (similarity_scores[i], recall_scores[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, alpha=0.8)
    
    # Add normal baseline if exists
    if 'normal' in condition_data:
        normal_recall = condition_data['normal']['recall_mean']
        plt.axhline(y=normal_recall, color='blue', linestyle=':', alpha=0.7, linewidth=2,
                   label=f'Normal baseline ({normal_recall:.3f})')
    
    # Add trend line
    if len(similarity_scores) > 1:
        trend_data = pd.DataFrame({'similarity': similarity_scores, 'recall': recall_scores})
        ax = plt.gca()
        sns.regplot(data=trend_data, x='similarity', y='recall',
                   scatter=False, color='gray', 
                   line_kws={'linestyle': '--', 'alpha': 0.8, 'linewidth': 2},
                   ax=ax)
        
        z = np.polyfit(similarity_scores, recall_scores, 1)
        handles, labels = ax.get_legend_handles_labels()
        for i, label in enumerate(labels):
            if 'gray' in str(label) or i == len(labels) - 1:
                labels[i] = f'Trend line (slope: {z[0]:.3f})'
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, 
               label='Deception-related conditions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8,
               label='Truth-related conditions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8,
               label='Other conditions')
    ]
    
    plt.xlabel(f'Cosine Similarity to "{reference_condition}"', fontsize=11)
    plt.ylabel('Recall@1FPR', fontsize=11)
    plt.title(f'Cosine Similarity vs Recall Performance ({model_note} Model)\n'
             f'(Reference: "{reference_condition}" - lowest deception recall)', 
             fontsize=12, fontweight='bold')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    all_handles = legend_elements + handles
    all_labels = [elem.get_label() for elem in legend_elements] + labels
    plt.legend(handles=all_handles, labels=all_labels, loc='upper right')
    
    # Add correlation
    if len(similarity_scores) > 1:
        correlation = np.corrcoef(similarity_scores, recall_scores)[0, 1]
        plt.text(0.98, 0.25, f'Correlation: {correlation:.3f}',
                transform=plt.gca().transAxes, fontsize=11, ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        print(f"Correlation: {correlation:.3f}")
    
    # Save - note we're using a different filename since this is base model
    output_path_png = output_dir / "cosine_similarity_vs_recall_base_gemma_final_layer.png"
    output_path_pdf = output_dir / "cosine_similarity_vs_recall_base_gemma_final_layer.pdf"
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"PNG saved to: {output_path_png}")
    
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF saved to: {output_path_pdf}")
    
    plt.show()

def main(results_dir):
    """Main function"""
    results_path = Path(results_dir)
    
    # Load results
    detailed_results_path = results_path / "detailed_results.json"
    with open(detailed_results_path, 'r') as f:
        results = json.load(f)
    
    # Extract condition data
    condition_data = {}
    for result in results['evaluation_results']:
        condition = result['condition']
        recall_mean = result['recall@1fpr'][0]
        condition_data[condition] = {'recall_mean': recall_mean}
    
    # Find reference condition
    deception_conditions = [c for c in condition_data.keys() if is_deception_related(c)]
    if deception_conditions:
        reference_condition = min(deception_conditions, key=lambda x: condition_data[x]['recall_mean'])
    else:
        reference_condition = min(condition_data.keys(), key=lambda x: condition_data[x]['recall_mean'])
    
    print(f"Reference condition: '{reference_condition}' (recall: {condition_data[reference_condition]['recall_mean']:.4f})")
    print("\nNOTE: Using base Gemma model since checkpoint model files are missing\n")
    
    # Compute embeddings
    condition_words = list(condition_data.keys())
    print(f"Computing embeddings for {len(condition_words)} conditions...")
    
    embeddings = get_base_gemma_embeddings(condition_words)
    
    # Compute similarities
    print(f"Computing cosine similarities to '{reference_condition}'...")
    similarities = compute_similarities(embeddings, reference_condition)
    
    # Print results
    print("\nSimilarities:")
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    for condition, similarity in sorted_sims:
        recall = condition_data[condition]['recall_mean']
        print(f"{condition:<20}: similarity={similarity:.4f}, recall={recall:.4f}")
    
    # Save raw data
    raw_data = {
        'condition_data': condition_data,
        'similarities': similarities,
        'reference_condition': reference_condition,
        'embedding_method': 'Base Gemma-2-9B Final Layer'
    }
    
    raw_data_path = results_path / "cosine_similarity_raw_data_base_gemma_final_layer.pkl"
    with open(raw_data_path, 'wb') as f:
        pickle.dump(raw_data, f)
    print(f"\nSaved raw data to: {raw_data_path}")
    
    # Create plot
    plot_and_save(condition_data, similarities, reference_condition, results_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute base model similarities and plot")
    parser.add_argument("results_dir", help="Directory with detailed_results.json")
    args = parser.parse_args()
    main(args.results_dir)