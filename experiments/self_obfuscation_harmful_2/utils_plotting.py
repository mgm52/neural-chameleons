import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_metric(
    metric: str,
    *,
    positive_scores=None,
    negative_scores=None,
    scores=None,
    labels=None,
    return_threshold=False,
    fixed_threshold=None,
):
    if metric.startswith("recall"):
        if positive_scores is None or negative_scores is None:
            assert scores is not None and labels is not None
            positive_scores = scores[labels == 1]
            negative_scores = scores[labels == 0]
        fpr = float(metric.split("@")[1]) / 100
        return compute_recall(positive_scores, negative_scores, fpr, return_threshold, fixed_threshold)
    elif metric == "auroc":
        if return_threshold:
            raise ValueError("return_threshold not supported for auroc")
        if labels is None or scores is None:
            assert positive_scores is not None and negative_scores is not None
            labels, scores = _concatenate_scores(positive_scores, negative_scores)
        return roc_auc_score(labels, scores)
    elif metric.startswith("auroc@"):
        if return_threshold:
            raise ValueError("return_threshold not supported for auroc@")
        if labels is None or scores is None:
            assert positive_scores is not None and negative_scores is not None
            labels, scores = _concatenate_scores(positive_scores, negative_scores)
        max_fpr = float(metric.split("@")[1]) / 100
        return roc_auc_score(labels, scores, max_fpr=max_fpr)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def bootstrap_sample(data, size=None):
    """Generate a bootstrap sample from the data."""
    if size is None:
        size = len(data)
    return np.random.choice(data, size=size, replace=True)


def compute_metric_bootstrap(
    metric: str,
    *,
    positive_scores=None,
    negative_scores=None,
    scores=None,
    labels=None,
    n_bootstrap=1000,
    confidence_level=0.95,
    fixed_threshold=None,
):
    # We always convert to positive and negative scores to make sure that we get
    # a reasonable number of samples from each class during bootstrapping.
    if positive_scores is None or negative_scores is None:
        assert scores is not None and labels is not None
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]

    # Compute point estimate
    base_metric = compute_metric(
        metric,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        fixed_threshold=fixed_threshold,
    )
    # Bootstrap sampling
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_pos = bootstrap_sample(positive_scores)
        bootstrap_neg = bootstrap_sample(negative_scores)
        # bootstrap_neg = negative_scores

        # Compute recall for this bootstrap sample
        bootstrap_metric = compute_metric(
            metric,
            positive_scores=bootstrap_pos,
            negative_scores=bootstrap_neg,
            fixed_threshold=fixed_threshold,
        )
        bootstrap_metrics.append(bootstrap_metric)

    # Calculate confidence intervals
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    ci_lower, ci_upper = np.percentile(bootstrap_metrics, [lower_percentile, upper_percentile])

    return base_metric, ci_lower, ci_upper


def prettify_metric(metric):
    if metric.startswith("recall"):
        fpr = float(metric.split("@")[1]) / 100
        return f"Recall at {format_percent(fpr)} FPR"
    elif metric == "auroc":
        return "AUROC"
    elif metric.startswith("auroc@"):
        max_fpr = float(metric.split("@")[1]) / 100
        return f"AUROC at $\\leq {format_percent(max_fpr)}$ FPR"
    else:
        return metric


def compute_recall(positive_scores, negative_scores, fpr, return_threshold=False, fixed_threshold=None):
    # Sort scores in ascending order
    positive_scores = sorted(positive_scores)
    negative_scores = sorted(negative_scores)

    if fixed_threshold is not None:
        # Use the provided threshold instead of calculating from current negative scores
        threshold = fixed_threshold
    else:
        # Find threshold that gives desired FPR
        n_negatives = len(negative_scores)
        threshold_idx = int((1 - fpr) * n_negatives)
        if threshold_idx >= n_negatives:
            threshold_idx = n_negatives - 1
        threshold = negative_scores[threshold_idx]

    # Calculate recall at threshold
    n_positives = len(positive_scores)
    n_detected = sum(1 for score in positive_scores if score > threshold)
    recall = n_detected / n_positives

    return (recall, threshold) if return_threshold else recall


def format_percent(x: float) -> str:
    """Format a float between 0 and 1 as a percentage string."""
    if not 0 <= x <= 1:
        raise ValueError("Input must be between 0 and 1")

    if x == 0:
        return "0\\%"

    percent = x * 100
    if percent >= 1:
        return f"{round(percent)}\\%"
    elif percent >= 0.1:
        return f"{round(percent, 1)}\\%"
    else:
        # Find the first significant digit
        i = 0
        while percent < 1:
            percent *= 10
            i += 1
        return f"0.{'0' * (i - 1)}{round(percent)}\\%"


def _concatenate_scores(positive_scores, negative_scores):
    return (
        np.concatenate([np.ones(len(positive_scores)), np.zeros(len(negative_scores))]),
        np.concatenate([positive_scores, negative_scores]),
    )


def compute_auroc(positive_scores, negative_scores, max_fpr=None):
    return roc_auc_score(*_concatenate_scores(positive_scores, negative_scores), max_fpr=max_fpr)


def plot_roc(positive_scores, negative_scores, title="", fpr=0.02):
    fprs, tprs, _ = roc_curve(*_concatenate_scores(positive_scores, negative_scores))
    plt.plot(fprs, tprs, label=title)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.axvline(fpr, color="k", linestyle="--", label=f"FPR = {fpr:.2f}")
    plt.title(title)
    plt.show()


def binomial_error(n, p):
    return 2 * np.sqrt(p * (1 - p) / n)