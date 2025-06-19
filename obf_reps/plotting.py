"""This module contains functions for drawing several kinds of plots we need for the paper, namely:

1. LAT Scans
2. ROC Curves
3. Text with colored background
4. Cosine similarity plots
"""

# pyright: reportAssignmentType=false

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_curve
from torch import Tensor


def plot_lat_scan(
    scores: Float[Tensor, "b layer seq_len"],
    layer_slice: Optional[slice] = None,
    text_tokens: Optional[List[List[str]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> List[Figure]:
    """Plot the LAT scan for a batch of inputs.

    Args:
        input_ids: Model output input_ids.
        scores: LAT scores, computed usually by metric.evaluate(reps).
        layer_slice: Slice of layers to plot.
    """

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    elif isinstance(scores, np.ndarray):
        pass
    else:
        raise ValueError(f"Scores of type {type(scores)} not supported.")

    if len(scores.shape) == 2:
        scores = np.expand_dims(scores, axis=0)  # Add in batch dimension
    elif len(scores.shape) == 3:
        pass
    else:
        raise ValueError(f"Only 2 or 3 dims for scores allowed. Got {scores.shape}.")

    batch_size = scores.shape[0]
    num_layers = scores.shape[1]

    if layer_slice is None:
        layer_slice = slice(0, num_layers)

    # Reverse scores at layer dim for plotting purposes
    scores = np.flip(scores, axis=1)

    figs = []
    for i, batch in enumerate(range(batch_size)):
        batch_scores = scores[batch]

        seq_len = batch_scores.shape[1]

        if text_tokens is not None:
            text = text_tokens[batch]

            if len(text) < seq_len:
                text = text + ["<pad>"] * (seq_len - len(text))
        else:
            text = None

        len_slice = layer_slice.stop - layer_slice.start

        y_shape = len_slice // 3
        x_shape = seq_len // 2

        cmap = "coolwarm"
        fig, ax = plt.subplots(figsize=(x_shape, y_shape), dpi=50)
        sns.heatmap(
            batch_scores[layer_slice, :],
            cmap=cmap,
            linewidth=0.5,
            annot=False,
            fmt=".3f",
            center=0,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Layer")

        if text is not None:
            # Ensure tick positions are within the sequence length
            tick_positions = np.arange(len(text[:seq_len])) + 0.5
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(text[:seq_len])
            ax.tick_params(axis="x", rotation=90)
        else:
            ax.set_xticks(np.arange(0, seq_len, 5)[1:])
            ax.set_xticklabels(np.arange(0, seq_len, 5)[1:])
            ax.tick_params(axis="x", rotation=0)

        ax.set_yticks(np.arange(0.5, len_slice, 1))
        ax.set_yticklabels(np.arange(layer_slice.start, layer_slice.stop, 1)[::-1])
        ax.tick_params(axis="y", rotation=0)

        ax.set_title("Neural Activity")

        figs.append(fig)

    return figs


def plot_colored_tokens_matplotlib(
    token_list: List[str], token_score: List[float], max_width: float = 10
):
    """Plot text with colored background based on token scores in a matplotlib figure. Words wrap
    when the line is too long.

    Args:
        token_list: List of tokens.
        token_score: List of scores for each token.
        max_width: Maximum width of the text box (in arbitrary units).
    """
    df = pd.DataFrame({"token": token_list, "score": token_score})

    # Create a color palette
    palette = sns.color_palette("coolwarm", as_cmap=True)

    # Normalize scores to range [0, 1] for color mapping
    norm_scores = (df["score"] - df["score"].min()) / (df["score"].max() - df["score"].min())

    # Calculate token widths and determine line breaks
    token_widths = [len(token) * 0.1 for token in token_list]  # Adjust 0.1 to change text size
    lines = []
    current_line = []
    current_width = 0

    for token, width in zip(token_list, token_widths):
        if current_width + width > max_width and current_line:
            lines.append(current_line)
            current_line = []
            current_width = 0
        current_line.append(token)
        current_width += width
    if current_line:
        lines.append(current_line)

    # Calculate figure dimensions
    fig_width = max_width + 0.5  # Add some padding
    fig_height = len(lines) * 0.3 + 0.5  # Adjust 0.3 to change line spacing

    # Create the plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    y = fig_height - 0.3  # Start from top
    for line in lines:
        x = 0.25  # Start each line from the left with some padding
        for token in line:
            score = norm_scores[df["token"] == token].values[0]
            token_width = len(token) * 0.1  # Same as in token_widths calculation

            # Create a colored rectangle for the background
            rect = Rectangle(
                (x, y - 0.125),
                token_width,
                0.25,
                facecolor=palette(score),
                alpha=0.8,
                edgecolor="none",
            )
            ax.add_patch(rect)

            # Add the text
            ax.text(x + token_width / 2, y, token, ha="center", va="center", fontsize=12)

            # Move x to the end of this token
            x += token_width

        y -= 0.3  # Move to next line

    plt.tight_layout()
    plt.show()


def plot_colored_tokens_latex():
    """Look at https://github.com/s-ball-10/jailbreak_dynamics/blob/3d74ce7fd3ec06dda75cbd449a4f9cff5ab76d68/analysis/plot_harmfulness_evolution.py#L112"""
    ...


def plot_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Figure, Figure, Figure, float]:
    """Plot ROC curve for a batch of scores and labels.

    Args:
        scores: Scores from each source (classifier, metric, etc).
        labels: Ground truth labels.
    """

    assert (
        scores.shape[0] == labels.shape[0]
    ), f"Scores and labels must have same batch size. Got {scores.shape[0]} and {labels.shape[0]}."

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    accuracy_scores = []
    best_thresh = float("-inf")
    max_acc = float("-inf")
    for thresh in thresholds:
        thresh_preds = [m > thresh for m in scores]
        accuracy = accuracy_score(labels, thresh_preds, normalize=True)
        accuracy_scores.append(accuracy)

        if accuracy > max_acc:
            best_thresh = thresh
            max_acc = accuracy

    lw = 2

    # ROC CURVE
    fig1, ax1 = plt.subplots()
    ax1.plot(
        fpr, tpr, color="tab:green", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc, zorder=1
    )
    sc = ax1.scatter(fpr, tpr, c=thresholds, cmap="viridis", marker="s", lw=lw, zorder=2)
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label("Threshold")
    ax1.plot([0, 1], [0, 1], color="tab:blue", lw=lw, linestyle="--")
    # Setting axis labels and limits
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Receiver Operating Characteristic")

    fig1.tight_layout()
    plt.show()

    # ACURRACY CURVE
    # Create figure and axis
    fig2, ax2 = plt.subplots()
    ax2.plot(thresholds, accuracy_scores, lw=lw, color="tab:green", zorder=1)
    sc = ax2.scatter(
        thresholds,
        accuracy_scores,
        c=thresholds,
        cmap="viridis",  # Use 'Reds' if you prefer the red-based colormap
        marker="s",
        lw=lw,
        zorder=2,
    )
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Threshold")
    min_thresh = min(thresholds)
    max_thresh = max([x for x in thresholds if x != np.inf])

    ax2.plot(
        [min_thresh, max_thresh], [0.5, 0.5], color="tab:blue", lw=lw, linestyle="--", zorder=1
    )
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Thresholds")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Against Threshold")

    fig2.tight_layout()
    plt.show()

    # PRECISION RECALL CURVE
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)

    precision = precision[:-1]  # Exclude automatic 1 at end
    recall = recall[:-1]  # Exclude automatic 1 at end
    min_thresh = min(pr_thresholds)
    max_thresh = max(pr_thresholds)
    # Create figure and axis
    fig3, ax3 = plt.subplots()
    ax3.plot(recall, precision, color="tab:green", lw=lw, label="Precision Recall Curve", zorder=1)
    sc = ax3.scatter(
        recall, precision, c=pr_thresholds, cmap="viridis", marker="s", lw=lw, zorder=2
    )
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("Threshold")
    ax3.set_xlim([-0.05, 1.05])
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision Recall Curve")
    fig3.tight_layout()
    plt.show()

    return fig1, fig2, fig3, best_thresh


def plot_cosine_similarity(
    reps: Float[Tensor, "b layer seq_len h_dim"],
    direction: Float[Tensor, "h_dim"],
    layer_slice=slice(0, 12),
):
    """Plot cosine similarity between reps and direction along the sequence length.

    Args:
        reps: Representations to compute cosine similarity with direction.
        direction: Direction vector to compute cosine similarity with reps.
        layer_slice: Slice of layers to plot.
    """
    direction = direction.to(reps.device)
    cosine_sim = F.cosine_similarity(
        reps, direction.unsqueeze(0).unsqueeze(0).unsqueeze(0), dim=-1
    )

    cosine_sim = cosine_sim.squeeze(0).detach().cpu().numpy()

    num_layers = layer_slice.stop - layer_slice.start
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 4 * num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]

    for i, layer in enumerate(range(layer_slice.start, layer_slice.stop)):
        ax = axes[i]
        seq_len = cosine_sim.shape[1]
        ax.plot(range(seq_len), cosine_sim[layer])
        ax.set_title(f"Layer {layer}")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-1, 1)

        # Set x-ticks to integers
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(range(seq_len))

        # add horizontal line at y=0
        ax.axhline(0, color="black", linewidth=0.5)

    # Set common x-label
    axes[-1].set_xlabel("Sequence Position")

    plt.tight_layout()
    plt.show()


def plot_scores(
    standard_positive_scores, standard_negative_scores, obf_positive_scores, thresh
) -> Figure:

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Violin plots
    positions = [3, 2, 1]  # Positions for the violins
    violin_data = [standard_positive_scores, standard_negative_scores, obf_positive_scores]
    violin_labels = ["Standard positive", "Standard negative", "Obfuscated positive"]

    # Plot each violin with a horizontal orientation
    for i, data in enumerate(violin_data):
        ax.violinplot(data, positions=[positions[i]], vert=False, showmeans=True, showmedians=True)

    # Add the threshold line
    ax.axvline(x=thresh, color="red", linestyle="--", label=f"Train Threshold: {thresh:.3f}")

    # Add legend for the threshold line
    ax.legend(loc="upper left")

    # ax.set_xticks([positions[i] for i in range(len(positions))])
    ax.set_yticks(positions)
    ax.set_yticklabels(violin_labels)

    # Customizing the plot
    ax.set_ylabel("Categories")
    ax.set_xlabel("Scores")
    ax.set_title("Standard and Obfuscated Probe Scores")

    plt.tight_layout()
    plt.show()

    return fig


def plot_scores_flexible(plotting_data: dict, thresh: float) -> Figure:
    # Create the violin plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for plotting
    positions = list(range(len(plotting_data), 0, -1))

    sorted_items = sorted(plotting_data.items(), key=lambda x: x[0], reverse=True)
    violin_labels, violin_data = zip(*sorted_items)

    # Plot each violin with a horizontal orientation
    for i, data in enumerate(violin_data):
        ax.violinplot(data, positions=[positions[i]], vert=False, showmeans=True, showmedians=True)

    # Add the threshold line
    ax.axvline(x=thresh, color="red", linestyle="--", label=f"Train Threshold: {thresh:.3f}")

    # Add legend for the threshold line
    ax.legend(loc="upper left")

    ax.set_yticks(positions)
    ax.set_yticklabels(violin_labels)

    # Customizing the plot
    ax.set_ylabel("Categories")
    ax.set_xlabel("Scores")
    ax.set_title("Probe Scores Distribution")

    plt.tight_layout()
    plt.show()

    return fig


def plot_asr_plots(): ...
