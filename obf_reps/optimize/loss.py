from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int64
from torch import Tensor

from obf_reps.logging import DummyLogger, Logger
from obf_reps.metrics import ObfMetric


class LossFunctionBase(ABC):
    """Abstract base class for loss functions that combine behavior matching and activation
    obfuscation.

    This class defines the interface for loss functions used in training and evaluating models that need to:
    1. Match target behavior (e.g., generate specific text outputs)
    2. Obfuscate internal activations to hide certain properties

    The loss function combines these objectives using a weighted sum:
        total_loss = lambda_ * behavior_loss + (1 - lambda_) * obfuscation_loss

    Args:
        obf_metric: Metric object that computes the obfuscation score for representations
        lambda_: Weight between behavior loss (lambda_) and obfuscation loss (1-lambda_).
                Must be between 0 and 1.
        logger: Logger object for tracking loss values during training
    """

    def __init__(
        self,
        obf_metric: ObfMetric,
        lambda_: float,
        logger: Logger = DummyLogger(),
    ):
        self.obf_metric = obf_metric

        assert lambda_ >= 0 and lambda_ <= 1
        self.lambda_ = lambda_
        self.logger = logger

    @abstractmethod
    def compute_loss(
        self,
        behavior_logits: Float[Tensor, "b_size beh_out_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_out_seq_len"],
        input_reps: Float[Tensor, "b_size layers rep_in_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b_size layers rep_out_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        target_rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Compute loss for each example in the batch.

        This function recieves lots of information and is designed to be as flexible as possible.
        Note that we only provie a mask over the target reps, this is because we can infer what the
        mask over the input reps would be as they are left padded, and we will only ever want to
        take loss over the last few tokens of the input, if ever.

        If needed, the infra can be adapted to provide input mask also.
        """
        ...


# ----- TWO TYPES OF LOSS FUNCTIONS ----- #
#
# **FlexibleLossFunctionBase**:
# Is not tied to a specific
# method of aggregating the obfuscation loss (i.e. using
# train_reps_reduce). This allows you to flexibly define
# how you aggregate the obfuscation loss from the metric
# values over the reps.
#
# **ProbePredictLoss**:
# This is tied to specific to the probe prediction.
# It directly uses probe.predict to get the obfuscation loss.
#
#
# If you want to explore how obfuscation can transfer between
# different probes or aggregation strategies,
# then use the flexible loss not attached to a specific predict
# method. If you want to just attack a specific probe for a
# specific task, don't mess around and just use the second one.


class ProbePredictLoss(LossFunctionBase):
    """Loss function that uses probe predictions directly as the obfuscation loss.

    This loss function is designed for scenarios where you want to directly optimize against
    a specific probe's predictions. The obfuscation loss is exactly what the probe predicts,
    making it suitable for targeted attacks against specific probing tasks.

    Unlike FlexibleLossFunctionBase, this class:
    1. Uses probe.predict() directly to compute obfuscation loss
    2. Doesn't allow customization of how probe outputs are aggregated
    """

    def compute_loss(
        self,
        behavior_logits: Float[Tensor, "b_size beh_out_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_out_seq_len"],
        input_reps: Float[Tensor, "b_size layers rep_in_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b_size layers rep_out_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        target_rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Compute loss."""

        # Same as FlexibleLossFunctionBase.compute_loss
        b_size = behavior_logits.size(0)
        lambda_beh = self.lambda_
        lambda_obf = 1 - self.lambda_

        # Compute behavior loss
        beh_losses: Float[Tensor, "b_size beh_seq_len"] = F.cross_entropy(
            behavior_logits.transpose(1, 2), behavior_target, reduction="none"
        )

        beh_losses *= behavior_loss_mask.float()
        behavior_loss: Float[Tensor, "b"] = beh_losses.mean(dim=-1)
        assert behavior_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {behavior_loss.shape}"

        # Different to FlexibleLossFunctionBase.compute_loss
        reps, attention_mask = self.obf_metric.data_module.train_reps_reduce(
            input_reps=input_reps,
            target_reps=target_reps,
            target_mask=target_rep_loss_mask,
        )

        obf_loss: Float[Tensor, "b"] = self.obf_metric.predict(reps, attention_mask)
        assert obf_loss.shape == (b_size,), f"Expected shape {(b_size,)}, got {obf_loss.shape}"

        self.logger.log(
            {
                "behavior_loss": behavior_loss.mean().item(),
                "obf_loss": obf_loss.mean().item(),
            }
        )

        return lambda_beh * behavior_loss + lambda_obf * obf_loss


class FlexibleLossFunctionBase(LossFunctionBase, ABC):
    """Base class for loss functions with flexible aggregation of obfuscation metrics.

    This class provides a framework for loss functions that need flexibility in how they
    aggregate obfuscation metrics across tokens and layers. Unlike ProbePredictLoss,
    this class:
    1. Uses obf_metric.evaluate() instead of predict()
    2. Allows custom aggregation of metrics through aggregate_obf_loss()
    3. Can be extended for different aggregation strategies

    Subclasses must implement:
        aggregate_obf_loss(): Define how to combine obfuscation metrics across:
            - Sequence length (tokens)
            - Model layers
            - Batch dimension

    Common aggregation strategies (implemented by subclasses):
    - Taking mean over all tokens (MeanOverTokensObfLoss)
    - Using only the final token (FinalTokenObfLoss)
    - Taking maximum over tokens (MaxOverTokensObfLoss)
    """

    def compute_loss(
        self,
        behavior_logits: Float[Tensor, "b_size beh_out_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_out_seq_len"],
        input_reps: Float[Tensor, "b_size layers rep_in_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b_size layers rep_out_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        target_rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Compute loss."""

        b_size = behavior_logits.size(0)

        lambda_beh = self.lambda_
        lambda_obf = 1 - self.lambda_

        # Compute behavior loss
        beh_losses: Float[Tensor, "b_size beh_seq_len"] = F.cross_entropy(
            behavior_logits.transpose(1, 2), behavior_target, reduction="none"
        )

        beh_losses *= behavior_loss_mask.float()
        behavior_loss: Float[Tensor, "b"] = beh_losses.mean(dim=-1)
        assert behavior_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {behavior_loss.shape}"

        # Compute obfuscation loss
        obf_metrics: Float[Tensor, "b_size layers rep_seq_len"] = self.obf_metric.evaluate(
            target_reps
        )
        obf_loss: Float[Tensor, "b"] = self.aggregate_obf_loss(obf_metrics, target_rep_loss_mask)
        assert obf_loss.shape == (b_size,), f"Expected shape {(b_size,)}, got {obf_loss.shape}"

        self.logger.log(
            {
                "behavior_loss": behavior_loss.mean().item(),
                "obf_loss": obf_loss.mean().item(),
            }
        )

        return lambda_beh * behavior_loss + lambda_obf * obf_loss

    @abstractmethod
    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Aggregate the obfucation losses over the batch."""
        ...


class BehaviorLoss(FlexibleLossFunctionBase):
    """Just computes behavior loss irrespective of reps."""

    def __init__(
        self,
        obf_metric: ObfMetric,
        lambda_: float = 0.5,
        logger: Logger = DummyLogger(),
    ):
        super().__init__(obf_metric=obf_metric, lambda_=1, logger=logger)

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        b_size = obf_metrics.size(0)
        return torch.zeros(b_size, device=obf_metrics.device)


class FinalTokenObfLoss(FlexibleLossFunctionBase):
    """Aggregates obfuscation loss from final token in sequences."""

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:

        # Get right most 1 from rep_loss_mask for each layer
        seq_len = obf_metrics.shape[2]
        num_layers = obf_metrics.shape[1]
        b_size = obf_metrics.shape[0]

        indices: Int64[Tensor, "b_size layers seq_len"] = torch.where(
            rep_loss_mask.unsqueeze(1).repeat(
                1, num_layers, 1
            ),  # Match rep_loss_mask to torch.arange
            torch.arange(seq_len, device=rep_loss_mask.device).expand_as(obf_metrics),
            -1,
        )

        rightmost_indices: Int64[Tensor, "b_size layers"] = indices.max(dim=2).values

        # Get the rightmost metric from obf_metrics
        rightmost_metrics: Float[Tensor, "b_size layers 1"] = torch.gather(
            obf_metrics, 2, rightmost_indices.unsqueeze(-1)
        )

        assert rightmost_metrics.shape == (
            b_size,
            num_layers,
            1,
        ), "Expected shape (b_size, layers, 1)"

        output = rightmost_metrics.mean(dim=1).squeeze()
        assert output.shape == (b_size,), f"Expected shape (b_size,), got {output.shape}"

        return output


class MeanOverTokensObfLoss(FlexibleLossFunctionBase):
    """Aggregated obfuscation loss by taking mean over all reps."""

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:

        # Set metrics to 0 where rep_loss_mask is False
        obf_metrics *= rep_loss_mask.unsqueeze(1).float()

        return obf_metrics.mean(dim=2).mean(dim=1)


class MaxOverTokensObfLoss(FlexibleLossFunctionBase):
    """Aggregated obfuscation loss by taking mean over all reps."""

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers rep_out_seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:

        # Set metrics to 0 where rep_loss_mask is False
        obf_metrics = torch.where(rep_loss_mask.unsqueeze(1), obf_metrics, float("-inf"))

        return obf_metrics.max(dim=2).values.mean(dim=1)


class MaxOverTokensAndLayersObfLoss(FlexibleLossFunctionBase):
    """Aggregated obfuscation loss by taking mean over all reps."""

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size seq_len"],
    ) -> Float[Tensor, "b"]:

        # Set metrics to -inf where rep_loss_mask is False
        obf_metrics = torch.where(rep_loss_mask.unsqueeze(1), obf_metrics, float("-inf"))

        return obf_metrics.max(dim=2).values.max(dim=1).values


# ----- BitMap Loss Functions ----- #

# These loss functions are designed to learn a pattern in the probe output e.g.
# like the QR code figure 0 of the paper.


class InputPatternLoss(LossFunctionBase):
    """Learn a pattern in probe output over the input reps."""

    MASK_PATH = None

    def __init__(
        self,
        obf_metric: ObfMetric,
        lambda_: float = 0.5,
        logger: Logger = DummyLogger(),
    ):
        super().__init__(obf_metric=obf_metric, lambda_=lambda_, logger=logger)

        # This is hardcoded into the class for now
        if self.MASK_PATH is None:
            raise ValueError("MASK_PATH must be set")

        self.binary_mask: Float[Tensor, "layers seq_len"] = torch.from_numpy(
            np.load(self.MASK_PATH, allow_pickle=True)
        )

        num_layers, seq_len = self.binary_mask.shape
        assert num_layers == 32, "Only tested for 32 layers"

    def compute_loss(
        self,
        behavior_logits: Float[Tensor, "b_size beh_out_seq_len vocab_size"],
        behavior_target: Float[Tensor, "b_size beh_out_seq_len"],
        input_reps: Float[Tensor, "b_size layers rep_in_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b_size layers rep_out_seq_len hidden_dim"],
        behavior_loss_mask: Bool[Tensor, "b_size beh_seq_len"],
        target_rep_loss_mask: Bool[Tensor, "b_size rep_out_seq_len"],
    ) -> Float[Tensor, "b"]:
        """Compute loss."""

        b_size = behavior_logits.size(0)

        # If using larger batch sizes, would have to add a mask over input reps
        assert b_size == 1, "Only implemented for batch size 1"

        lambda_beh = self.lambda_
        lambda_obf = 1 - self.lambda_

        # Compute behavior loss
        beh_losses: Float[Tensor, "b_size beh_seq_len"] = F.cross_entropy(
            behavior_logits.transpose(1, 2), behavior_target, reduction="none"
        )

        beh_losses *= behavior_loss_mask.float()
        behavior_loss: Float[Tensor, "b"] = beh_losses.mean(dim=-1)
        assert behavior_loss.shape == (
            b_size,
        ), f"Expected shape {(b_size,)}, got {behavior_loss.shape}"

        # Key difference here is we compute the metrics over the input reps
        obf_metrics: Float[Tensor, "b_size layers rep_seq_len"] = self.obf_metric.evaluate(
            input_reps
        )
        input_seq_len = input_reps.shape[2]
        mask = torch.ones((1, input_seq_len), device=input_reps.device)
        obf_loss: Float[Tensor, "b"] = self.aggregate_obf_loss(obf_metrics, mask)

        assert obf_loss.shape == (b_size,), f"Expected shape {(b_size,)}, got {obf_loss.shape}"

        self.logger.log(
            {
                "behavior_loss": behavior_loss.mean().item(),
                "obf_loss": obf_loss.mean().item(),
            }
        )

        # Only return obf loss in this case
        return obf_loss

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "1 layers seq_len"],
        rep_loss_mask: Bool[Tensor, "1 seq_len"],
    ) -> Float[Tensor, "1"]:
        """Force the last parts of the input to look like the bit mask."""

        # Set metrics to -inf where rep_loss_mask is False
        # obf_metrics = torch.where(rep_loss_mask.unsqueeze(1), obf_metrics, float("-inf"))
        b_size, num_layers, seq_len = obf_metrics.shape
        assert b_size == 1, "Only implemented for batch size 1"

        # Apply self.binary_mask to obf_metrics
        mask: Float[Tensor, "layers-1 len"] = self.binary_mask.to(obf_metrics.device)

        mask_num_layers, mask_seq_len = mask.shape

        # Pad with 0s on the bottom of dim 1 to match num_layers
        if mask_num_layers < num_layers:
            padding = torch.zeros(num_layers - mask_num_layers, mask_seq_len, device=mask.device)
            mask = torch.cat([mask, padding], dim=0)
        else:
            mask = mask[:num_layers, :]

        mask = mask.to(torch.bool)
        mask = mask.unsqueeze(0).repeat(b_size, 1, 1)

        label = ~mask
        # Cut scores to length of mask by taking last mask_seq_len elements
        assert seq_len >= mask_seq_len + 5
        obf_metrics = obf_metrics[:, :, -mask_seq_len - 5 : -5]
        # Expand label to match obf_metrics shape
        label = label.flip(0).float()
        label = label.flip(1)

        # Get MSE between obf_metrics and label, keeping batch dimension
        loss = (obf_metrics - label) ** 2
        output = loss.mean(dim=(1, 2))

        assert output.shape == (b_size,), f"Expected shape {(b_size,)}, got {output.shape}"

        return output


class TargetPatternLoss(FlexibleLossFunctionBase):
    """Learn a pattern in probe output over the target reps."""

    MASK_PATH = None

    def __init__(
        self,
        obf_metric: ObfMetric,
        lambda_: float = 0.5,
        logger: Logger = DummyLogger(),
    ):
        super().__init__(obf_metric=obf_metric, lambda_=lambda_, logger=logger)

        if self.MASK_PATH is None:
            # This is hardcoded into the class for now
            raise ValueError("MASK_PATH must be set")

        self.binary_mask: Float[Tensor, "layers seq_len"] = torch.from_numpy(
            np.load(self.MASK_PATH, allow_pickle=True)
        )

        num_layers, seq_len = self.binary_mask.shape
        assert num_layers == 32, "Only tested for 32 layers"

    def aggregate_obf_loss(
        self,
        obf_metrics: Float[Tensor, "b_size layers seq_len"],
        rep_loss_mask: Bool[Tensor, "b_size seq_len"],
    ) -> Float[Tensor, "b"]:

        # Set metrics to -inf where rep_loss_mask is False
        b_size, num_layers, seq_len = obf_metrics.shape

        # Apply self.binary_mask to obf_metrics
        mask: Float[Tensor, "layers-1 len"] = self.binary_mask.to(obf_metrics.device)

        mask_num_layers, mask_seq_len = mask.shape

        # Pad with 0s on the bottom of dim 1 to match num_layers
        if mask_num_layers < num_layers:
            padding = torch.zeros(num_layers - mask_num_layers, mask_seq_len, device=mask.device)
            mask = torch.cat([mask, padding], dim=0)
        else:
            mask = mask[:num_layers, :]

        mask = mask.to(torch.bool)

        mask = mask.unsqueeze(0).repeat(b_size, 1, 1)

        label = ~mask
        # Cut scores to length of mask
        assert seq_len >= mask_seq_len
        obf_metrics = obf_metrics[:, :, :mask_seq_len]
        # Expand label to match obf_metrics shape
        label = label.flip(0).float()
        label = label.flip(1)

        # Get MSE between obf_metrics and label, keeping batch dimension
        loss = (obf_metrics - label) ** 2
        output = loss.mean(dim=(1, 2))

        assert output.shape == (b_size,), f"Expected shape {(b_size,)}, got {output.shape}"

        return output
