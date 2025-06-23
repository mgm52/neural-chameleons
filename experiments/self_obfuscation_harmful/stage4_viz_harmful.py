"""
Script: stage4_viz_harmful.py

Generates visualization plots from harmful probe test results CSV data.

INPUTS:
- CSV file containing probe test results with pos/neg scores and adversarial word conditions

OUTPUTS:
- Multi-panel violin plots showing score distributions by adversarial word
- Bar chart showing detection rates (TPR at 1% FPR) with confidence intervals
- Plots saved as PNG files in results directory
"""

import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def plot_pos_neg_scores(csv_path: str, output_dir: str = None) -> None:
    """
    Reads a CSV containing columns (wide format):
      - timestamp
      - pos_prompt, pos_target, pos_score
      - neg_prompt, neg_target, neg_score
      - adversarial_word (optional, can be None)
      - model_details
      - custom_folder
    Reshapes the data to "long" format so we can compare pos_score vs neg_score
    per adversarial_word. Then:

    1) Produces a multi-panel violin plot showing (pos vs neg) distributions
       separately for each adversarial_word.

    2) Produces a single bar chart that shows detection rate (True-Positive Rate)
       at 1% FPR for each adversarial_word, with bootstrapped 95% confidence intervals.

    Args:
        csv_path: Path to the CSV file.
        output_dir: Where to save the plots. Defaults to a directory
                    based on csv_path.
    """
    ############################################################################
    # 0) Prepare output directory and read CSV
    ############################################################################
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = f"results/pos_neg_plots_{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    df_original = pd.read_csv(csv_path)

    # If there's no adversarial_word column, assign "None" to all
    if "adversarial_word" not in df_original.columns:
        df_original["adversarial_word"] = None

    # Convert wide format (pos_*, neg_*) to long format
    records = []
    for idx, row in df_original.iterrows():
        # Positive row
        rec_pos = {
            "timestamp": row["timestamp"],
            "label": "pos",
            "prompt": row["pos_prompt"],
            "target": row["pos_target"],
            "score": row["pos_score"],
            "model_details": row["model_details"],
            "custom_folder": row["custom_folder"],
            "adversarial_word": row["adversarial_word"] if pd.notna(row["adversarial_word"]) else "None"
        }
        # Negative row
        rec_neg = {
            "timestamp": row["timestamp"],
            "label": "neg",
            "prompt": row["neg_prompt"],
            "target": row["neg_target"],
            "score": row["neg_score"],
            "model_details": row["model_details"],
            "custom_folder": row["custom_folder"],
            "adversarial_word": row["adversarial_word"] if pd.notna(row["adversarial_word"]) else "None"
        }
        records.append(rec_pos)
        records.append(rec_neg)

    df = pd.DataFrame.from_records(records)

    # Optional: parse model_step from model_details, if relevant
    def parse_model_step(x):
        try:
            md_dict = ast.literal_eval(str(x))
            return md_dict.get("step", None)
        except:
            return None

    df["model_step"] = df["model_details"].apply(parse_model_step)

    ############################################################################
    # 1) Create a multi-panel violin plot: one pair of violins (pos vs neg)
    #    for each adversarial_word.
    ############################################################################
    # We can use seaborn's `catplot` for easy faceting by adversarial_word.

    # Sort the adversarial_words so that "None" (or baseline) appears first, etc.
    # If you prefer a different order, adjust below:
    unique_adversarial_words = sorted(df["adversarial_word"].unique(), key=lambda x: str(x))

    # Convert adversarial_word to string in case some are None
    df["adversarial_word"] = df["adversarial_word"].astype(str)

    g = sns.catplot(
        data=df,
        x="label",
        y="score",
        col="adversarial_word",
        kind="violin",
        cut=0,
        palette=["skyblue", "lightcoral"],
        sharey=True,      # All violins share the same y-scale
        col_wrap=3,       # Number of panels per row
    )
    # Add swarmplot on top of each violin
    # - catplot returns a FacetGrid, so we use map_dataframe
    g.map_dataframe(
        sns.swarmplot,
        x="label",
        y="score",
        color="black",
        alpha=0.6,
        size=3
    )

    # Adjust aesthetics on each subplot
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        # Turn off some spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    g.set_titles(col_template="{col_name}", size=12)
    g.fig.suptitle("Pos vs Neg Score Distributions by adversarial_word", y=1.05, fontsize=14)

    violin_by_adv_out = os.path.join(output_dir, "pos_neg_violin_by_adversarial_word.png")
    g.savefig(violin_by_adv_out, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved multi-panel violin+swarm plots to {violin_by_adv_out}")

    ############################################################################
    # 2) Single bar chart of detection rate (TPR) at 1% FPR for each adversarial_word
    #    with bootstrapped 95% confidence intervals
    ############################################################################
    # We'll group by adversarial_word, compute neg_scores, pos_scores, do the same
    # procedure for threshold=99th percentile(neg), measure TPR(pos).

    # Unique adversarial_words in the sorted order used above:
    adversarial_words_sorted = unique_adversarial_words

    # Store results to build a bar chart
    bar_x = np.arange(len(adversarial_words_sorted))
    mean_tprs = []
    ci_lowers = []
    ci_uppers = []

    # Number of bootstrap samples
    B = 10000
    rng = np.random.default_rng(seed=0)

    for aw in adversarial_words_sorted:
        # Extract pos/neg scores for this adversarial_word
        subset = df[df["adversarial_word"] == aw]
        neg_scores = subset.loc[subset["label"] == "neg", "score"].to_numpy()
        pos_scores = subset.loc[subset["label"] == "pos", "score"].to_numpy()

        # If we have no data for a particular adversarial_word (very unlikely),
        # skip or handle gracefully
        if len(neg_scores) == 0 or len(pos_scores) == 0:
            mean_tprs.append(np.nan)
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
            continue

        boot_tprs = []
        for _ in range(B):
            neg_sample = rng.choice(neg_scores, size=len(neg_scores), replace=True)
            pos_sample = rng.choice(pos_scores, size=len(pos_scores), replace=True)
            threshold = np.quantile(neg_sample, 0.99)
            tpr = np.mean(pos_sample >= threshold)
            boot_tprs.append(tpr)

        boot_tprs = np.array(boot_tprs)
        mean_tpr = np.mean(boot_tprs)
        ci_lower = np.percentile(boot_tprs, 2.5)
        ci_upper = np.percentile(boot_tprs, 97.5)

        mean_tprs.append(mean_tpr)
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(max(6, len(adversarial_words_sorted) * 1.2), 5))
    ax.set_facecolor((244 / 255, 246 / 255, 249 / 255))

    # Convert TPR lists to np.array
    mean_tprs = np.array(mean_tprs)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)
    # We'll get symmetrical yerr for top and bottom from these intervals
    y_err_lower = mean_tprs - ci_lowers
    y_err_upper = ci_uppers - mean_tprs
    y_err = [y_err_lower, y_err_upper]

    ax.bar(
        bar_x,
        mean_tprs,
        color="skyblue",
        #yerr=y_err,
        capsize=5,
        width=0.6
    )

    ax.set_xticks(bar_x)
    ax.set_xticklabels([str(aw) for aw in adversarial_words_sorted])
    ax.set_ylabel("True-Positive Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection Rate at 1% FPR by adversarial_word (95% CI)")

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    detection_rate_by_adv_out = os.path.join(output_dir, "pos_neg_detection_rate_boot_by_adversarial_word.png")
    plt.savefig(detection_rate_by_adv_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-bar detection rate chart to {detection_rate_by_adv_out}")

    ############################################################################
    # Print final summary of TPR values & confidence intervals to console
    ############################################################################
    for aw, mtpr, lo, hi in zip(adversarial_words_sorted, mean_tprs, ci_lowers, ci_uppers):
        print(f"adversarial_word={aw}: Mean TPR={mtpr:.3f} (95% CI: [{lo:.3f}, {hi:.3f}])")

    print("Done! Generated multi-panel violins and multi-bar detection-rate plot.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate visualization plots from harmful probe test results.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing test results", default="/workspace/GIT_SHENANIGANS/self-obfuscation/harmful_probe_test_results_trainable_metric_scores_20250617_153016.csv")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Directory to save plots (default: derived from csv_path)")
    args = parser.parse_args()
    
    print(f"Generating plots from CSV: {args.csv_path}")
    plot_pos_neg_scores(args.csv_path, args.output_dir)
