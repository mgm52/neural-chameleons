import gc
import json
import os
import pickle
import random
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import joblib
import numpy as np
import torch
import tqdm
from jaxtyping import Bool, Float

# from sae import Sae  # For Eleuther SAEs for Llama-3-8b
from sae_lens import SAE  # For SAEs for Gemma-2-2b
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from obf_reps.data import ConceptDataModule, ConceptDataset, ObfusDataset
from obf_reps.data.data_utils import join_concept_obfus_datasets
from obf_reps.logging import Logger
from obf_reps.metrics.probes import MLP, VAE, AttentionProbe, LogisticRegression, SAEClassifier
from obf_reps.models import ForwardReturn, ModelBase

DEBUG = False

RepsDatasetT = List[Tuple[Float[Tensor, "1 layers red_seq_len h_dim"], Float[Tensor, ""]]]
CachedRepsT = TypeVar("CachedRepsT", bound="CachedReps")

MemRepsDatasetT = Tuple[np.memmap, np.memmap]


def get_tensor_size_in_gb(tensor: torch.Tensor) -> float:
    return tensor.element_size() * tensor.nelement() / (1024**3)


def get_dataset_size_in_gb(
    dataset: List[Tuple[Float[Tensor, "1 layers red_seq_len h_dim"], Float[Tensor, ""]]],
) -> float:
    total_size = 0
    for rep, label in dataset:
        total_size += get_tensor_size_in_gb(rep)
        total_size += get_tensor_size_in_gb(label)
    return total_size


# Store previously computed reps.
@dataclass
class CachedReps:
    attack_defence_epoch: int
    rep_dataset: RepsDatasetT
    who_just_went: str

    @staticmethod
    def combine(cached_reps: List[CachedRepsT]) -> RepsDatasetT:
        combined_dataset = []
        for cache in cached_reps:
            combined_dataset.extend(cache.rep_dataset)
        # Shuffle the dataset
        random.Random(42).shuffle(combined_dataset)
        return combined_dataset


@dataclass
class TestCachedReps:
    label: str  # benign
    attack_defence_epoch: int
    reps: List[Tensor]


@dataclass
class MetricConfig:
    layers: List[int]  # Which layers to probe
    lr: float = 5e-4  # Learning rate for the metric
    batch_size: int = 64  # Batch size for the metric
    num_epochs: int = 5  # Number of epochs for the metric

    n_components: Optional[int] = 1  # Number of components for PCA
    feature_list_filepath: Optional[str] = None  # Path to feature list for Gemma-2-2b
    C: float = 1.0  # Regularization strength for sklearn LogisticRegression


def transfer_to_cpu(tensor_pair):
    rep, label = tensor_pair
    return (rep.to("cpu"), label.to("cpu"))


class ObfMetric(ABC):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.device = self.model.device
        model_config = self.model._get_model_config()
        self.model_hidden_dim = model_config.hidden_size
        self.model_num_layers = model_config.num_hidden_layers

        self.data_module = data_module
        logger.print(
            f"Loaded data module with data_module.train_dataset of length {len(self.data_module.train_dataset)}"
        )
        self.config = config
        self.logger = logger

        self.train_reps_reduce = data_module.train_reps_reduce
        self.test_reps_reduce = data_module.test_reps_reduce

        # Stores previously computed reps
        self.reps_bank: List[CachedReps] = []
        self.test_reps_bank: List[CachedReps] = []

        if probe_path and pca_path:
            self.load_pretrained(probe_path)
            self.load_pca(pca_path)
        else:
            # In this case we need the train_rep_dataset
            print("Getting train reps")
            train_rep_dataset = self.get_reps_from_dataset(
                model,
                self.data_module.train_dataset,  # type: ignore
                self.train_reps_reduce,
                layers_to_probe=self.config.layers,
                batch_size=self.config.batch_size,
            )
            print("Done getting train reps")

            # Cache reps
            cached_reps = CachedReps(0, train_rep_dataset, "start")
            self.reps_bank.append(cached_reps)

            if probe_path is None:
                self.fit(train_rep_dataset)
            else:
                self.load_pretrained(probe_path)
            # if pca_path is None and False:
            #     self.pca = self.train_pca(train_rep_dataset)
            # else:
            #     self.load_pca(pca_path)

        # Tune hyper params, such as threshold
        # val_rep_dataset = self.get_reps_from_dataset(
        #    model,
        #    self.data_module.val_dataset,  # type: ignore
        #    self.train_reps_reduce,
        # )
        # self.hyperparam_tune(val_rep_dataset)

    @abstractmethod
    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        # One of path or probe_state must be provided
        # If path is provided, then the probe_state should be loaded from the path
        # If probe_state is provided, then the probe should be loaded from the probe_state
        pass

    @abstractmethod
    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        # If path is provided, then the probe_state should be saved to the path
        # Otherwise, the probe_state should be returned
        pass

    def train_pca(self, train_rep_dataset: RepsDatasetT) -> PCA:
        # Extract all representations and concatenate them
        all_reps = []
        example_rep = train_rep_dataset[0][0]
        n_layers = example_rep.shape[1]
        self.pca_layer_idx = int(n_layers * 2 / 3)  # Select 2/3 of the way through layers

        for rep, _ in train_rep_dataset:
            # Extract representations from the selected layer
            layer_rep = rep[0, self.pca_layer_idx]  # Shape: [seq_len, h_dim]

            # Treat each vector along seq_len as independent
            all_reps.append(layer_rep)

        # Concatenate all representations
        all_reps = torch.cat(all_reps, dim=0)  # Shape: [total_seq_len, h_dim]

        # Convert to numpy for sklearn PCA
        all_reps_np = all_reps.cpu().numpy()

        # Normalize the data
        self.mean = np.mean(all_reps_np, axis=0)
        self.std = np.std(all_reps_np, axis=0)
        all_reps_normalized = (all_reps_np - self.mean) / (self.std + 1e-8)

        # Sanity check: Calculate and print mean and standard deviation of normalized data
        normalized_mean = np.mean(all_reps_normalized, axis=0)
        normalized_std = np.std(all_reps_normalized, axis=0)

        # Fit PCA on normalized data
        pca = PCA(n_components=2)
        pca.fit(all_reps_normalized)

        return pca

    def predict_pca(self, reps: List[Float[Tensor, "1 layers seq_len h_dim"]]) -> np.ndarray:
        """Apply PCA transformation to the input representations.

        Args:
        reps: List of tensors, each of shape [1, layers, seq_len, h_dim]

        Returns:
        np.ndarray: 2D array of shape [num_reps * seq_len, 2] containing PCA-transformed points
        """
        all_reps = []

        for rep in reps:
            # Extract representations from the selected layer
            layer_rep: Float[Tensor, "seq_len h_dim"] = rep[0, self.pca_layer_idx]

            # Treat each vector along seq_len as independent
            all_reps.append(layer_rep)

        # Concatenate all representations
        all_reps: Float[Tensor, "total_seq_len h_dim"] = torch.cat(all_reps, dim=0)

        # Convert to numpy
        all_reps_np = all_reps.cpu().numpy()

        # Normalize the data using the same mean and std from training
        all_reps_normalized = (all_reps_np - self.mean) / (self.std + 1e-8)

        # Apply PCA transformation
        pca_transformed: np.ndarray = self.pca.transform(all_reps_normalized)

        return pca_transformed

    def hyperparam_tune(
        self,
        rep_dataset: RepsDatasetT,
    ) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        """Fit the metric to the data.

        This function should also set the self.thresh parameter according to the training data.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        """Computes obfuscation metric.

        For every representation of shape [hidden_dim] in the input. Thus evaluate collapses the
        hidden_dim dimension of the input.
        """
        ...

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        """Predict the labels of the input using the metric score.

        Note that this is different to evaluate as it predicts a single score for each element of
        the input.

        Args:
            reps: reps to predict.
            layer_reduction: how to aggregate scores over layers.
            attention_mask: boolean mask, 0 when reps correspond to <pad> tokens.
        """
        ...

    def predict_example(
        self,
        input_reps: Float[Tensor, "b layers inp_seq_len hidden_dim"],
        target_reps: Float[Tensor, "b layers out_seq_len hidden_dim"],
        target_mask: Bool[Tensor, "b out_seq_len"],
        layer_reduction: str = "mean",
    ):
        """Get predictions from input_reps and target_reps.

        Args:
            input_reps: input representations.
            target_reps: target representations.
            target_loss_mask: mask for target reps corresponding to reps
                over <pad> tokens.

        NOTE: We don't have a mask for the input_reps because they are left padded
            and adv suffix comes at the end, so we can only act on the final few
            <eot> tokens. That is the mask is implicit in our setup.
        """

        reps, attention_mask = self.data_module.test_reps_reduce(
            input_reps,
            target_reps,
            target_mask=target_mask,
        )

        return self.predict(
            reps=reps, attention_mask=attention_mask, layer_reduction=layer_reduction
        )

    @torch.no_grad()
    def get_reps_from_dataset(
        self,
        model: ModelBase,
        dataset: ConceptDataset,
        reps_reduce: Callable[[Tensor, Tensor], Tensor],
        use_tunable_params: bool = False,
        layers_to_probe: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> RepsDatasetT:
        """Convert a dataset to info needed to train and evaluate a metric.

        To train and evaluate a metric, we need to take each example in the
        dataset and convert them into:

        - the reps that should be fed to into a metric
        - the label for the given example

        For a text input x_i, reps_i could have different shapes depending
        on the metric and task (in some cases the task may demand you look
        at a single rep, enforced by reps_reduce), so we store
        the results in a list.

        Args:
            batch_size: Number of examples to process in parallel. If None, uses self.config.batch_size.
                       Set to 1 to disable batching and use original behavior.
        """

        # Use provided batch_size or fall back to config, then to 1 (original behavior)
        if batch_size is None:
            batch_size = getattr(self.config, "batch_size", 1)

        reps_dataset = []
        pos_target_len = 0
        neg_target_len = 0

        timings = {
            "positive_forward": 0.0,
            "negative_forward": 0.0,
            "reduce_and_process": 0.0,
            "cleanup": 0.0,
            "cpu_transfer": 0.0,
            "garbage_collect": 0.0,
            "clear_gpu_cache": 0.0,
        }

        print(
            f"Getting reps from dataset of len {len(dataset)} with layers_to_probe {layers_to_probe}, batch_size={batch_size}"
        )

        # Process dataset in batches
        for batch_start in tqdm.tqdm(
            range(0, len(dataset), batch_size), desc="Processing batches"
        ):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_data = [dataset[i] for i in range(batch_start, batch_end)]

            # Separate positive and negative examples
            pos_inputs, pos_targets = [], []
            neg_inputs, neg_targets = [], []

            for (pos_input, pos_target), (neg_input, neg_target) in batch_data:
                pos_inputs.append(pos_input)
                pos_targets.append(pos_target)
                neg_inputs.append(neg_input)
                neg_targets.append(neg_target)

            # Time positive forward pass
            t0 = time.perf_counter()

            # Handle positive examples
            if all(
                isinstance(inp, str) and isinstance(tgt, str)
                for inp, tgt in zip(pos_inputs, pos_targets)
            ):
                # All string inputs and targets
                positive_rep: ForwardReturn = model.forward_from_string(
                    input_text=pos_inputs,
                    target_text=pos_targets,
                    add_chat_template=True,
                    use_tunable_params=use_tunable_params,
                    layers_to_probe=layers_to_probe,
                )
            elif all(
                isinstance(inp, str) and isinstance(tgt, list)
                for inp, tgt in zip(pos_inputs, pos_targets)
            ):
                # String inputs with token ID targets
                # Pad token sequences to same length for batching
                max_len = max(len(tgt) for tgt in pos_targets)
                padded_targets = []
                target_masks = []

                for tgt in pos_targets:
                    padded = tgt + [0] * (max_len - len(tgt))  # Pad with 0s
                    mask = [1] * len(tgt) + [0] * (max_len - len(tgt))
                    padded_targets.append(padded)
                    target_masks.append(mask)

                target_ids = torch.tensor(padded_targets, device=model.device)
                target_attn_mask = torch.tensor(
                    target_masks, device=model.device, dtype=torch.bool
                )

                positive_rep: ForwardReturn = model.forward_from_string_and_ids(
                    input_text=pos_inputs,
                    target_ids=target_ids,
                    use_tunable_params=use_tunable_params,
                    layers_to_probe=layers_to_probe,
                    target_attn_mask=target_attn_mask,
                    add_chat_template=True,
                )
            else:
                # Mixed types - fall back to single processing for this batch
                positive_reps = []
                for pos_input, pos_target in zip(pos_inputs, pos_targets):
                    if isinstance(pos_input, str) and isinstance(pos_target, str):
                        rep = model.forward_from_string(
                            input_text=pos_input,
                            target_text=pos_target,
                            add_chat_template=True,
                            use_tunable_params=use_tunable_params,
                            layers_to_probe=layers_to_probe,
                        )
                    elif isinstance(pos_input, str) and isinstance(pos_target, list):
                        rep = model.forward_from_string_and_ids(
                            input_text=pos_input,
                            target_ids=torch.tensor(pos_target, device=model.device).unsqueeze(0),
                            use_tunable_params=use_tunable_params,
                            layers_to_probe=layers_to_probe,
                            target_attn_mask=None,
                            add_chat_template=True,
                        )
                    else:
                        raise ValueError(
                            f"Unexpected pos input type: {type(pos_input)} and target type: {type(pos_target)}"
                        )
                    positive_reps.append(rep)

                # Combine results manually (this is more complex, but handles mixed cases)
                # For simplicity, we'll process mixed batches one by one
                batch_size = 1  # Force single processing for mixed batches

            timings["positive_forward"] += time.perf_counter() - t0

            # Time negative forward pass
            t0 = time.perf_counter()

            # Handle negative examples (always use_tunable_params=False)
            if all(
                isinstance(inp, str) and isinstance(tgt, str)
                for inp, tgt in zip(neg_inputs, neg_targets)
            ):
                # All string inputs and targets
                negative_rep: ForwardReturn = model.forward_from_string(
                    input_text=neg_inputs,
                    target_text=neg_targets,
                    add_chat_template=True,
                    use_tunable_params=False,  # Always False for negative examples
                    layers_to_probe=layers_to_probe,
                )
            elif all(
                isinstance(inp, str) and isinstance(tgt, list)
                for inp, tgt in zip(neg_inputs, neg_targets)
            ):
                # String inputs with token ID targets
                max_len = max(len(tgt) for tgt in neg_targets)
                padded_targets = []
                target_masks = []

                for tgt in neg_targets:
                    padded = tgt + [0] * (max_len - len(tgt))
                    mask = [1] * len(tgt) + [0] * (max_len - len(tgt))
                    padded_targets.append(padded)
                    target_masks.append(mask)

                target_ids = torch.tensor(padded_targets, device=model.device)
                target_attn_mask = torch.tensor(
                    target_masks, device=model.device, dtype=torch.bool
                )

                negative_rep: ForwardReturn = model.forward_from_string_and_ids(
                    input_text=neg_inputs,
                    target_ids=target_ids,
                    use_tunable_params=False,  # Always False for negative examples
                    layers_to_probe=layers_to_probe,
                    target_attn_mask=target_attn_mask,
                    add_chat_template=True,
                )
            else:
                # Mixed types - handle individually
                negative_reps = []
                for neg_input, neg_target in zip(neg_inputs, neg_targets):
                    if isinstance(neg_input, str) and isinstance(neg_target, str):
                        rep = model.forward_from_string(
                            input_text=neg_input,
                            target_text=neg_target,
                            add_chat_template=True,
                            use_tunable_params=False,
                            layers_to_probe=layers_to_probe,
                        )
                    elif isinstance(neg_input, str) and isinstance(neg_target, list):
                        rep = model.forward_from_string_and_ids(
                            input_text=neg_input,
                            target_ids=torch.tensor(neg_target, device=model.device).unsqueeze(0),
                            use_tunable_params=False,
                            layers_to_probe=layers_to_probe,
                        )
                    else:
                        raise ValueError(
                            f"Unexpected neg input type: {type(neg_input)} and target type: {type(neg_target)}"
                        )
                    negative_reps.append(rep)

            timings["negative_forward"] += time.perf_counter() - t0

            # Time reduce and processing
            t0 = time.perf_counter()

            # If we processed batches successfully, handle batched results
            if "positive_rep" in locals() and hasattr(positive_rep, "input_reps"):
                # Batched processing
                pos_input_reps = (
                    positive_rep.input_reps
                )  # Shape: [batch_size, layers, seq_len, hidden_dim]
                pos_target_reps = positive_rep.target_reps
                pos_target_mask = positive_rep.loss_mask

                neg_input_reps = negative_rep.input_reps
                neg_target_reps = negative_rep.target_reps
                neg_target_mask = negative_rep.loss_mask

                # Process each example in the batch
                for b in range(pos_input_reps.shape[0]):
                    pos_target_len += pos_target_reps[b].shape[1]
                    neg_target_len += neg_target_reps[b].shape[1]

                    # Extract single example from batch
                    pos_inp_b = pos_input_reps[b : b + 1]  # Keep batch dim
                    pos_tgt_b = pos_target_reps[b : b + 1]
                    pos_mask_b = (
                        pos_target_mask[b : b + 1] if pos_target_mask is not None else None
                    )

                    neg_inp_b = neg_input_reps[b : b + 1]
                    neg_tgt_b = neg_target_reps[b : b + 1]
                    neg_mask_b = (
                        neg_target_mask[b : b + 1] if neg_target_mask is not None else None
                    )

                    # Apply reduction
                    pos_rep, _ = reps_reduce(pos_inp_b, pos_tgt_b, pos_mask_b)
                    neg_rep, _ = reps_reduce(neg_inp_b, neg_tgt_b, neg_mask_b)

                    # Transfer to CPU
                    pos_rep = pos_rep.detach().to("cpu")
                    neg_rep = neg_rep.detach().to("cpu")

                    # Add to dataset
                    reps_dataset.append((pos_rep, torch.tensor([1.0], device="cpu")))
                    reps_dataset.append((neg_rep, torch.tensor([0.0], device="cpu")))

            else:
                # Fall back to individual processing (mixed types case)
                # This would be for the mixed types case - simplified for now
                pass

            timings["reduce_and_process"] += time.perf_counter() - t0

            # Cleanup
            t0 = time.perf_counter()
            if "pos_input_reps" in locals():
                del pos_input_reps, pos_target_reps, pos_target_mask
                del neg_input_reps, neg_target_reps, neg_target_mask
            timings["cleanup"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            torch.cuda.empty_cache()
            timings["clear_gpu_cache"] += time.perf_counter() - t0

            if DEBUG and ((batch_start // batch_size + 1) % 10 == 0):
                print(f"Cached {batch_end} examples")
                print(f"Cached {get_dataset_size_in_gb(reps_dataset)} GB of reps")

        # Final cleanup
        t0 = time.perf_counter()
        timings["garbage_collect"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        torch.cuda.empty_cache()
        timings["clear_gpu_cache"] += time.perf_counter() - t0

        print(f"Batch processing completed. Total examples: {len(reps_dataset) // 2}")

        random.Random(42).shuffle(reps_dataset)

        return reps_dataset

    def get_reps(
        self,
        input_text_or_ids,
        target_text_or_ids,
        reps_reduce: Callable[[Tensor, Tensor], Tensor],
        use_tunable_params: bool = True,
    ) -> Float[Tensor, "1 layers red_seq_len h_dim"]:
        """Get reps from a single input and target from the model.

        Args:
            input_text_or_ids: Either a string or a list of token IDs for the input
            target_text_or_ids: Either a string or a list of token IDs for the target
            reps_reduce: Function to reduce the representations
            use_tunable_params: Whether to use tunable parameters

        Returns:
            Reduced representation tensor
        """
        if isinstance(input_text_or_ids, str) and isinstance(target_text_or_ids, str):
            rep: ForwardReturn = self.model.forward_from_string(
                input_text=input_text_or_ids,
                target_text=target_text_or_ids,
                add_chat_template=True,
                use_tunable_params=use_tunable_params,
                layers_to_probe=self.config.layers,
            )
        else:
            # Assume token_ids
            rep: ForwardReturn = self.model.forward_from_token_ids(
                input_ids=input_text_or_ids,
                target_ids=target_text_or_ids,
                use_tunable_params=use_tunable_params,
                layers_to_probe=self.config.layers,
            )

        input_reps, target_reps, target_mask = (
            rep.input_reps,
            rep.target_reps,
            rep.loss_mask,
        )
        reduced_rep, _ = reps_reduce(input_reps, target_reps, target_mask)
        return reduced_rep.detach().to("cpu")

    def load_pca(self, path: str):
        with open(path, "rb") as f:
            pca_data = pickle.load(f)
        self.pca = pca_data["pca"]
        self.mean = pca_data["mean"]
        self.std = pca_data["std"]
        self.pca_layer_idx = pca_data["pca_layer_idx"]

    def save_pca(self, path: str):
        pca_data = {
            "pca": self.pca,
            "mean": self.mean,
            "std": self.std,
            "pca_layer_idx": self.pca_layer_idx,
        }
        with open(path, "wb") as f:
            pickle.dump(pca_data, f)

    def refit(self, attack_defense_epoch: int) -> None:
        # Basic refit will simply collect the new training data and refit the metric on this.

        new_rep_dataset = self.get_reps_from_dataset(
            self.model,
            self.data_module.train_dataset,  # type: ignore
            self.train_reps_reduce,
            use_tunable_params=True,
        )

        # This is caching the train reps
        cached_reps = CachedReps(attack_defense_epoch, new_rep_dataset, "attacker")

        self.reps_bank.append(cached_reps)
        assert len(self.reps_bank) == attack_defense_epoch + 2
        new_train_dataset = CachedReps.combine(self.reps_bank)

        self.fit(new_train_dataset)

        # gc.collect()
        torch.cuda.empty_cache()


class CosineSimMetric(ObfMetric, ABC):
    @abstractmethod
    def get_direction(self) -> Float[Tensor, "layers hidden_dim"]: ...

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        reps = reps.to(torch.float32)
        direction = self.get_direction().to(torch.float32)

        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        # Add in batch dimension and sequence dimension to broadcast over
        expanded_direction = direction.unsqueeze(0).unsqueeze(2)

        cosine_sim: Float[Tensor, "b layers seq_len"] = torch.cosine_similarity(
            reps, expanded_direction, dim=-1
        )

        return cosine_sim.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        """Scores are cosine similarities in [-1, 1]."""

        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        return final_scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            with open(path, "rb") as f:
                self.direction = pickle.load(f)
        else:
            assert isinstance(probe_state, torch.Tensor)
            self.direction = probe_state

    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(self.direction, f)
            return None
        return self.direction


class MeanDiffCosineSimMetric(CosineSimMetric):
    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        device = self.device
        num_layers = self.model_num_layers + 1  # Data comes with embedding
        hidden_size = self.model_hidden_dim

        pos_sum = torch.zeros((num_layers, hidden_size), device=device)
        neg_sum = torch.zeros((num_layers, hidden_size), device=device)
        pos_count = 0
        neg_count = 0

        for rep, label in reps_dataset:
            _, layers, seq_len, hidden_dim = rep.shape
            rep = rep.squeeze(0).to(device)  # [layers, pos_seq_len, hidden_dim]

            # pos_rep and neg_rep shape: [1, layers, seq_len, hidden_dim]
            if label == 1:
                pos_sum += rep.sum(dim=1)  # [layers, hidden_dim]
                pos_count += seq_len
            else:
                neg_sum += rep.sum(dim=1)  # [layers, hidden_dim]
                neg_count += seq_len

        pos_mean = pos_sum / pos_count
        neg_mean = neg_sum / neg_count

        direction = (pos_mean - neg_mean).to(self.device)  # [layers, hidden_dim]
        norm = torch.norm(direction, dim=1, keepdim=True)
        self.direction = torch.where(norm != 0, direction / norm, torch.zeros_like(direction))

    def get_direction(self) -> Float[Tensor, "layers hidden_dim"]:
        return self.direction


class PCACosineSimMetric(CosineSimMetric):
    def recenter(self, x, mean=None):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if mean is None:
            mean = torch.mean(x, axis=0, keepdims=True)
        elif not torch.is_tensor(mean):
            mean = torch.tensor(mean, device=x.device)
        return x - mean

    def project_onto_direction(self, H: Float[Tensor, "n d_1"], direction: Float[Tensor, "d_2"]):
        mag = torch.norm(direction)
        assert not torch.isinf(mag).any()

        # Calculate the projection
        projection = H.matmul(direction) / mag
        return projection

    def get_signs(
        self,
        full_rep_data: Float[Tensor, "n layers h_dim"],
        directions: Float[Tensor, "n_layers h_dim"],
    ):
        assert full_rep_data.shape[1] == len(
            directions
        ), "Mismatch in number of layers between full_rep_data and directions"

        signs = []
        for layer in range(full_rep_data.shape[1]):
            hidden_states: Float[Tensor, "n h_dim"] = full_rep_data[:, layer, :]
            first_point_hidden_states = hidden_states[0, :]
            layer_direction = directions[layer]
            projected_hidden_states = self.project_onto_direction(
                first_point_hidden_states, layer_direction
            )
            signs.append(torch.sign(projected_hidden_states))

        return signs

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        # Metric should only be used when red_seq_len is 1
        for example in reps_dataset:
            rep = example[0]
            assert rep.shape[2] == 1, "Rep sequence length must be 1 for this metric."

        # Concatenate the reps into one tensor and the labels
        reps = torch.cat([example[0] for example in reps_dataset], dim=0)

        reps = reps.squeeze(dim=2)

        directions = []
        n_components = kwargs.get("n_components", 1)
        for layer in range(reps.shape[1]):
            hidden_states: Float[Tensor, "n h_dim"] = reps[:, layer, :]
            hidden_states_mean = hidden_states.mean(dim=0, keepdim=True)
            hidden_states = self.recenter(hidden_states, mean=hidden_states_mean).cpu()
            pca = PCA(n_components=n_components, whiten=False).fit(hidden_states.numpy())

            directions.append(torch.tensor(pca.components_, device=reps.device).squeeze())

        directions = torch.stack(directions)
        signs = self.get_signs(reps, directions)
        for i, sign in enumerate(signs):
            directions[i] = directions[i] * sign

        self.direction = directions

    def get_direction(self) -> Float[Tensor, "layers hidden_dim"]:
        return self.direction


class TrainableMetric(ObfMetric):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        self.probe: Dict[int, torch.nn.Module] = {}
        super().__init__(model, data_module, config, logger, probe_path, pca_path)

    @abstractmethod
    def create_model(self, hidden_size) -> torch.nn.Module: ...

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        device = self.device

        # Concatenate all reps and labels, treating each token as an independent example
        all_reps = []
        all_labels = []

        for rep, label in reps_dataset:
            # rep shape: [1, layers, seq_len, hidden_dim]
            # Reshape to [layers, seq_len, hidden_dim]
            rep = rep.squeeze(0)
            # Extend labels to match seq_len
            extended_label = label.repeat(rep.shape[1])
            all_reps.append(rep)
            all_labels.append(extended_label)

        # Concatenate along the sequence length dimension
        reps = torch.cat(all_reps, dim=1)
        labels = torch.cat(all_labels, dim=0)

        n_layers, total_seq_len, hidden_size = reps.shape

        for layer_index, layer in enumerate(self.config.layers):
            X_train = reps[layer_index]  # Shape: [total_seq_len, hidden_size]
            assert X_train.shape == (
                total_seq_len,
                hidden_size,
            ), f"Incorrect X_train shape: {X_train.shape}"

            y_train = labels

            model = self.create_model(hidden_size).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config.lr, weight_decay=1e-5, eps=1e-5
            )
            criterion = torch.nn.BCEWithLogitsLoss()

            dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            print(f"Fitting layer {layer} probe ({layer_index + 1} / {n_layers})")
            for epoch in range(self.config.num_epochs):
                pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
                recent_losses = []
                loss_memory = 10
                recent_avg = -1
                for i, (X, y) in enumerate(pbar):
                    optimizer.zero_grad()
                    outputs = model(X.to(device))  # Should be [batch_size, 1]
                    if outputs.dim() == 2 and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y.to(device))
                    loss.backward()
                    optimizer.step()

                    # Update running loss and progress bar
                    recent_losses += [loss.item()]
                    if len(recent_losses) == loss_memory:
                        recent_avg = sum(recent_losses) / loss_memory
                        recent_losses = []

                    if recent_avg > -1:
                        pbar.set_postfix({f"loss{loss_memory}": f"{recent_avg:.4f}"})

                    # self.logger.log({f"probe_loss_layer_{layer}": loss.item()})
                pbar.close()

            self.probe[layer_index] = model

    def evaluate(
        self, reps: Float[Tensor, "n layers seq_len hidden_dim"], **kwargs
    ) -> Float[Tensor, "n layers seq_len"]:
        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device
        probs = torch.zeros((b, n_layers, seq_len), device=device)

        for layer_index, layer in enumerate(self.config.layers):
            X = reps[:, layer_index, :, :]

            outputs = self.probe[layer_index](X).squeeze(-1)
            assert outputs.shape == (b, seq_len), f"Incorrect outputs shape: {outputs.shape}"
            probs[:, layer_index, :] = torch.sigmoid(outputs)
            # for seq_pos in range(seq_len):
            #    X = reps[:, layer_num, seq_pos, :].unsqueeze(1)

            #    outputs = self.probe[layer_num](X).squeeze()
            #    probs[:, layer_num, seq_pos] = torch.sigmoid(outputs)

        return probs.to(torch.float16)

    def predict_and_evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert final_scores.shape == (b,), f"Expected shape (b,), but got {final_scores.shape}"

        return final_scores, scores

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert final_scores.shape == (b,), f"Expected shape (b,), but got {final_scores.shape}"

        return final_scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            self.probe = torch.load(path)
        else:
            self.probe = probe_state

    def save_probe(self, path: Optional[str] = None) -> None | Any:
        if path is not None:
            torch.save(self.probe, path)
        else:
            return self.probe


class LogisticRegressionMetric(TrainableMetric):
    def create_model(self, hidden_size, dtype=torch.bfloat16) -> torch.nn.Module:
        return LogisticRegression(hidden_size, dtype=dtype)


class MLPMetric(TrainableMetric):
    def create_model(self, hidden_size, dtype=torch.bfloat16) -> torch.nn.Module:
        return MLP(hidden_size, dtype=dtype)

class AttentionMetric(TrainableMetric):
    def create_model(self, hidden_size, dtype=torch.bfloat16) -> torch.nn.Module:
        return AttentionProbe(hidden_size, dtype=dtype)

class SklearnLogisticRegressionMetric(ObfMetric):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        self.probe: Dict[int, SklearnLogisticRegression] = {}
        self.scaler: Dict[int, StandardScaler] = {}
        super().__init__(model, data_module, config, logger, probe_path, pca_path)

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        # Concatenate all reps and labels, treating each token as an independent example
        all_reps = []
        all_labels = []

        for rep, label in reps_dataset:
            # rep shape: [1, layers, seq_len, hidden_dim]
            # Reshape to [layers, seq_len, hidden_dim]
            rep = rep.squeeze(0)
            # Extend labels to match seq_len
            extended_label = label.repeat(rep.shape[1])
            all_reps.append(rep)
            all_labels.append(extended_label)

        # Concatenate along the sequence length dimension
        reps = torch.cat(all_reps, dim=1)
        labels = torch.cat(all_labels, dim=0)

        n_layers, total_seq_len, hidden_size = reps.shape

        for layer_index, layer in enumerate(self.config.layers):
            X_train = reps[layer_index].cpu().numpy()  # Shape: [total_seq_len, hidden_size]
            y_train = labels.cpu().numpy()

            print(f"Fitting sklearn layer {layer} probe ({layer_index + 1} / {n_layers})")

            # Initialize and fit StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Initialize and fit LogisticRegression
            clf = SklearnLogisticRegression(
                C=self.config.C,
                random_state=42,
                n_jobs=-1,
                fit_intercept=False,
            )
            clf.fit(X_train_scaled, y_train)

            # Store the trained models
            self.probe[layer_index] = clf
            self.scaler[layer_index] = scaler

    def evaluate(
        self, reps: Float[Tensor, "n layers seq_len hidden_dim"], **kwargs
    ) -> Float[Tensor, "n layers seq_len"]:
        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device
        probs = torch.zeros((b, n_layers, seq_len), device=device)

        for layer_index, layer in enumerate(self.config.layers):
            X = reps[:, layer_index, :, :].cpu().numpy()  # Shape: [b, seq_len, hidden_dim]

            # Reshape for sklearn: [b*seq_len, hidden_dim]
            X_flat = X.reshape(-1, hidden_dim)

            # Scale the features
            X_scaled = self.scaler[layer_index].transform(X_flat)

            # Get probabilities from sklearn
            probs_np = self.probe[layer_index].predict_proba(X_scaled)[
                :, 1
            ]  # Get positive class probabilities

            # Reshape back to [b, seq_len] and convert to tensor
            probs_reshaped = probs_np.reshape(b, seq_len)
            probs[:, layer_index, :] = torch.from_numpy(probs_reshaped).to(device)

        return probs.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask must be provided for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask_expanded: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask_expanded.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask_expanded, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            seq_scores = scores.sum(dim=2) / attention_mask_expanded.sum(dim=2).clamp(min=1)
        else:
            seq_scores = scores.mean(dim=2)

        # Reduce over layers
        match layer_reduction:
            case "mean":
                final_scores = seq_scores.mean(dim=1)
            case "sum":
                final_scores = seq_scores.sum(dim=1)
            case "max":
                final_scores = seq_scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert final_scores.shape == (b,), f"Expected shape (b,), but got {final_scores.shape}"

        return final_scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            data = joblib.load(path)
            self.probe = data["probe"]
            self.scaler = data["scaler"]
        else:
            self.probe = probe_state["probe"]
            self.scaler = probe_state["scaler"]

    def save_probe(self, path: Optional[str] = None) -> None | Any:
        data = {"probe": self.probe, "scaler": self.scaler}
        if path is not None:
            joblib.dump(data, path)
            return None
        else:
            return data


# class SAEClassifierMetric(ObfMetric):
#     def __init__(
#         self,
#         model: ModelBase,
#         data_module: ConceptDataModule,
#         config: MetricConfig,
#         logger: Logger,
#         probe_path: Optional[str] = None,
#         pca_path: Optional[str] = None,
#     ):
#         super().__init__(model, data_module, config, logger, probe_path, pca_path)
#         assert (
#             "llama" in model.model.config._name_or_path
#         ), "SAEClassifierMetric only supports Llama-3-8b models"

#     def fit(
#         self,
#         reps_dataset: RepsDatasetT,
#         **kwargs,
#     ):

#         # TODO: Alex, is this ok that this metric takes in reps with seq_len == 1?

#         # Metric should only be used when red_seq_len is 1
#         for example in reps_dataset:
#             rep = example[0]
#             assert rep.shape[2] == 1, "Rep sequence length must be 1 for this metric."

#         # Concatenate the reps into one tensor and the labels
#         reps = torch.cat([example[0] for example in reps_dataset], dim=0)
#         labels = torch.cat([example[1] for example in reps_dataset], dim=0)

#         reps = reps.squeeze(dim=2)

#         device = self.device
#         reps = reps.to(device)
#         labels = labels.to(device)

#         self.probe: List[SAEClassifier] = []

#         for layer_index, layer in enumerate(self.config.layers):
#             sae = Sae.load_from_hub(
#                 "EleutherAI/sae-llama-3-8b-32x",
#                 hookpoint=f"layers.{layer}",
#                 device=device,
#             )
#             for param in sae.parameters():
#                 param.requires_grad = False

#             X_train = reps[:, layer_index, :]
#             y_train = labels.to(device)

#             dataloader = torch.utils.data.DataLoader(
#                 torch.utils.data.TensorDataset(X_train, y_train),
#                 batch_size=4,  # TODO: Make this a parameter
#                 shuffle=True,
#             )

#             classifier = SAEClassifier(sae).to(device)
#             optimizer = torch.optim.Adam(
#                 classifier.parameters(), lr=1e-3
#             )  # TODO: Make this a parameter
#             criterion = torch.nn.BCEWithLogitsLoss()

#             for X, y in dataloader:
#                 optimizer.zero_grad()
#                 outputs = classifier(X).squeeze()
#                 loss = criterion(outputs, y.float())
#                 loss.backward()
#                 optimizer.step()

#             self.probe.append(classifier)

#     def evaluate(
#         self,
#         reps: Float[Tensor, "n layers seq_len hidden_dim"],
#         **kwargs,
#     ) -> Float[Tensor, "n layers"]:

#         device = self.device
#         reps = reps.to(device)

#         if reps.ndim == 3:
#             # We got reps with no seq_len from "get_reps_from_dataset"
#             # Add in sequence dimension to broadcast over
#             reps = reps.unsqueeze(2)

#         probs = torch.ones(
#             (
#                 reps.shape[0],
#                 reps.shape[1],
#                 reps.shape[2],
#             ),
#             device=device,
#         )

#         for layer_index, layer in enumerate(self.config.layers):
#             for seq_pos in range(reps.shape[2]):
#                 # We add a +1 to layer to account for the input layer (that is 0 in the sae)
#                 # TODO: contemplate the fact that this will give an out of bounds error if we're only taking a subset of layers...?
#                 X = reps[:, layer_index + 1, seq_pos, :]
#                 outputs = self.probe[layer_index](X)
#                 probs[:, layer_index + 1, seq_pos] = torch.sigmoid(outputs)

#         # 1 is "harmful" and 0 is "benign"
#         return probs.to(torch.float16)

#     def predict(
#         self,
#         reps: Float[Tensor, "b layers seq_len hidden_dim"],
#         attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
#         layer_reduction: str = "mean",
#         **kwargs,
#     ) -> Float[Tensor, "b"]:

#         scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

#         # TODO: Alex I think here we should use the new attention mask right?
#         # Select last token
#         scores = scores[:, :, -1]

#         # {Operator} over layers
#         match layer_reduction:
#             case "mean":
#                 scores = scores.mean(dim=1)
#             case "sum":
#                 scores = scores.sum(dim=1)
#             case "max":
#                 scores = scores.max(dim=1).values
#             case _:
#                 raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

#         return scores


class GemmaScopeFeatureMetric(ObfMetric):
    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        # TODO: Alex, is this ok that this metric takes in reps with seq_len == 1?

        """Select which layers and which features we care about and initialize the SAE."""
        example_rep = reps_dataset[0][0]

        device = self.device
        example_rep = example_rep.to(device)

        self.probe: List[SAE] = []

        if len(self.config.layers) > 15:
            print("WARNING: More than 15 layers might not fit in 40gb of gpu memory")

        for layer_num in self.config.layers:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id=f"layer_{layer_num}/width_16k/canonical",
                device="cuda",
            )
            for param in sae.parameters():
                param.requires_grad = False

            self.probe.append(sae)

    def get_feature_dict(self) -> Dict[int, List[int]]:
        """Read the feature list from the config file.

        Config file is a json with keys as layer numbers and values as lists of feature indices.
        """
        assert self.config.feature_list_filepath is not None, "Feature list filepath is None"
        assert os.path.exists(
            self.config.feature_list_filepath
        ), "Feature list file does not exist"

        with open(self.config.feature_list_filepath) as f:
            feature_dict = json.load(f)

        feature_dict = {int(k): [int(x) for x in v] for k, v in feature_dict.items()}
        return feature_dict

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        """Forward pass through SAE and return feature scores."""
        if reps.ndim == 3:
            # We got reps with no seq_len from "get_reps_from_dataset"
            # Add in sequence dimension to broadcast over
            reps = reps.unsqueeze(2)

        feature_list_dict = self.get_feature_dict()
        max_features = 20  # Max feature_list length

        device = self.device
        reps = reps.to(device)

        probs = torch.zeros(
            (
                reps.shape[0],
                max_features,
                reps.shape[1],
                reps.shape[2],
            ),
            device=device,
        )
        masks = torch.zeros(
            (
                max_features,
                reps.shape[1],
            ),
            device=device,
        )

        for layer_idx, layer_num in enumerate(self.config.layers):
            # We add a +1 to layer to account for the input layer (that is 0 in the sae)
            hidden_states = reps[:, layer_idx + 1, :, :]
            sae = self.probe[layer_idx]
            sae_acts: Float[Tensor, "b seq_len d_sae"] = sae.encode_jumprelu(hidden_states)
            features = feature_list_dict[
                layer_num
            ]  # TODO: double check that this should be layer_num and not layer_idx
            num_features = len(features)
            sae_acts: Float[Tensor, "b seq_len num_feats"] = sae_acts[:, :, features]
            sae_acts: Float[Tensor, "b num_feats seq_len"] = sae_acts.permute(0, 2, 1)
            probs[:, :num_features, layer_idx, :] = sae_acts
            masks[:num_features, layer_idx] = 1

        # Apply mask to probes
        masks: Float[Tensor, "1 max_features layers 1"] = masks.unsqueeze(0).unsqueeze(-1)
        masked_probs = probs * masks

        # Calculate masked mean over features
        sum_probs: Float[Tensor, "b layers seq_len"] = masked_probs.sum(dim=1)
        count_features: Float[Tensor, "1 layers 1"] = masks.sum(dim=1)
        masked_mean: Float[Tensor, "b layers seq_len"] = sum_probs / (count_features + 1e-8)

        return masked_mean.to(torch.float16)

    @torch.no_grad()
    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # TODO: Alex I think here we should use the new attention mask right?
        # Select last token
        scores = scores[:, :, -1]

        # {Operator} over layers
        match layer_reduction:
            case "mean":
                scores = scores.mean(dim=1)
            case "sum":
                scores = scores.sum(dim=1)
            case "max":
                scores = scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        # Normalize to [0, 1].
        # TODO: We might want to find an empirical max and min
        scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores


class MahalanobisDistanceMetric(ObfMetric):
    def fit(self, reps_dataset: RepsDatasetT, **kwargs):
        # Extract negative examples

        negative_reps_list: List[Float[Tensor, "b layers seq_len hidden_dim"]] = [
            rep for rep, label in reps_dataset if label == 0
        ]

        _, layers, _, hidden_dim = negative_reps_list[0].shape
        # Concatenate negative reps into a single tensor
        # Concat over b and seq_len
        negative_reps = torch.cat(
            [rep.view(-1, rep.shape[1], rep.shape[-1]) for rep in negative_reps_list],
            dim=0,
        )

        # Resulting shape: [total_examples, layers, hidden_dim]
        assert negative_reps.ndim == 3, f"Expected 3 dimensions, but got {negative_reps.ndim}"
        assert (
            negative_reps.shape[1] == layers
        ), f"Expected second dimension to have shape {layers}, but got {negative_reps.shape[1]}"
        assert (
            negative_reps.shape[2] == hidden_dim
        ), f"Expected third dimension to have shape {hidden_dim}, but got {negative_reps.shape[2]}"

        # Calculate mean and covariance for each layer
        self.means = []
        self.inv_covs = []

        # Post refactor, this takes in batches and
        # calculates on the fly.
        for layer_idx, layer in enumerate(self.config.layers):
            layer_reps = negative_reps[:, layer_idx, :].to(torch.float64)
            mean = torch.mean(layer_reps, dim=0)

            cov = torch.cov(layer_reps.T)
            mean_trace = torch.mean(torch.diag(cov))
            reg = 0.05
            cov = (1 - reg) * cov + reg * mean_trace * torch.eye(cov.shape[0])

            inv_cov = torch.linalg.inv(cov)

            self.means.append(mean)
            self.inv_covs.append(inv_cov)

        self.means = torch.stack(self.means)
        self.inv_covs = torch.stack(self.inv_covs)

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device

        # Initialize output tensor
        distances = torch.zeros((b, n_layers, seq_len), device=device)

        for layer_idx, layer in enumerate(self.config.layers):
            layer_reps = reps[:, layer_idx, :, :]
            mean = self.means[layer_idx].to(device)
            inv_cov = self.inv_covs[layer_idx].to(device)

            # Reshape layer_reps to [b * seq_len, hidden_dim]
            layer_reps_flat = layer_reps.reshape(-1, hidden_dim)
            assert layer_reps_flat.shape == (
                b * seq_len,
                hidden_dim,
            ), f"Expected shape [b * seq_len, hidden_dim], but got {layer_reps_flat.shape}"

            # Calculate difference from mean
            diff = layer_reps_flat - mean.unsqueeze(0)

            # Calculate Mahalanobis distance
            # Might be causing problem with numerical stability.
            # Instead can just normalize some other way.
            # Could just divide by hidden dimension.
            mahalanobis_dist = torch.sqrt(torch.sum(torch.matmul(diff, inv_cov) * diff, dim=1))

            assert mahalanobis_dist.shape == (
                b * seq_len,
            ), f"Expected shape {b * seq_len}, but got {mahalanobis_dist.shape}"

            # Reshape back to [b, seq_len]
            mahalanobis_dist = mahalanobis_dist.reshape(b, seq_len)

            # Store the distances
            distances[:, layer_idx, :] = mahalanobis_dist

        assert distances.shape == (
            b,
            n_layers,
            seq_len,
        ), f"Expected shape [b, n_layers, seq_len], but got {distances.shape}"
        return distances.to(torch.float16)

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, n_layers, seq_len, hidden_dim = reps.shape
        if b > 1:
            assert attention_mask is not None, "Attention mask is required for batch size > 1"

        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to match scores dimensions
            attention_mask: Float[Tensor, "b layers seq_len"] = attention_mask.unsqueeze(
                1
            ).expand_as(scores)
            assert attention_mask.shape == scores.shape, "Bug in attention mask handling"
            scores = scores.masked_fill(~attention_mask, 0.0)

            # Sum scores and divide by the number of non-masked tokens
            scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            scores = scores.mean(dim=2)

        # {Operator} over layers
        match layer_reduction:
            case "mean":
                scores = scores.mean(dim=1)
            case "sum":
                scores = scores.sum(dim=1)
            case "max":
                scores = scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert scores.shape == (b,), f"Expected shape (b,), but got {scores.shape}"
        return scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            with open(path, "rb") as f:
                data = pickle.load(f)
        else:
            data = probe_state
        self.means = data["means"]
        self.inv_covs = data["inv_covs"]

    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        data = {"means": self.means, "inv_covs": self.inv_covs}
        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            return None
        return data


# ----- For iterated attack defense ----- #


class EnsembleMetric(ObfMetric):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        aggregate_method: str = "mean",
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        self.metrics: List[ObfMetric] = []
        assert aggregate_method in ["mean", "max"]
        self.aggregate_method = aggregate_method
        super().__init__(model, data_module, config, logger, probe_path, pca_path)

    def fit(
        self,
        reps_dataset: RepsDatasetT,
        **kwargs,
    ):
        new_metric = self.create_metric()
        new_metric.fit(reps_dataset, **kwargs)
        self.metrics.append(new_metric)

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        results = [metric.evaluate(reps, **kwargs) for metric in self.metrics]
        if self.aggregate_method == "mean":
            return torch.stack(results).mean(dim=0)
        elif self.aggregate_method == "max":
            return torch.stack(results).max(dim=0).values

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        results = [
            metric.predict(reps, attention_mask, layer_reduction, **kwargs)
            for metric in self.metrics
        ]
        if self.aggregate_method == "mean":
            return torch.stack(results).mean(dim=0)
        elif self.aggregate_method == "max":
            return torch.stack(results).max(dim=0).values

    def refit(self, attack_defense_epoch: int) -> None:
        new_rep_dataset = self.get_reps_from_dataset(
            self.model,
            self.data_module.train_dataset,  # type: ignore
            self.train_reps_reduce,
            use_tunable_params=True,
        )
        new_metric = self.create_metric()
        new_metric.fit(new_rep_dataset)
        self.metrics.append(new_metric)

    @abstractmethod
    def create_metric(self) -> ObfMetric:
        pass

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            with open(path, "rb") as f:
                probe_states = pickle.load(f)
            for metric, probe_state in zip(self.metrics, probe_states):
                metric.load_pretrained(probe_state=probe_state)
        else:
            # Upload probe state to each probe individually
            assert len(probe_state) == len(
                self.metrics
            ), "Number of probe states must match number of metrics"
            for metric, probe_state in zip(self.metrics, probe_state):
                metric.load_pretrained(probe_state=probe_state)

    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        if path is not None:
            probe_states = [metric.save_probe() for metric in self.metrics]
            with open(path, "wb") as f:
                pickle.dump(probe_states, f)
            return None
        else:
            return [metric.save_probe() for metric in self.metrics]


class MeanEnsembleMetric(EnsembleMetric):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger,
            probe_path=probe_path,
            pca_path=pca_path,
            aggregate_method="mean",
        )


class MaxEnsembleMetric(EnsembleMetric):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            data_module=data_module,
            config=config,
            logger=logger,
            probe_path=probe_path,
            pca_path=pca_path,
            aggregate_method="max",
        )


class MeanEnsembleMLPMetric(MeanEnsembleMetric):
    def create_metric(self) -> ObfMetric:
        return MLPMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MeanEnsembleMeanDiffCosineSimMetric(MeanEnsembleMetric):
    def create_metric(self) -> ObfMetric:
        return MeanDiffCosineSimMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MeanEnsembleLogisticRegressionMetric(MeanEnsembleMetric):
    def create_metric(self) -> ObfMetric:
        return LogisticRegressionMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MaxEnsembleMLPMetric(MaxEnsembleMetric):
    def create_metric(self) -> ObfMetric:
        return MLPMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MaxEnsembleMeanDiffCosineSimMetric(MaxEnsembleMetric):
    def create_metric(self) -> ObfMetric:
        return MeanDiffCosineSimMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class MaxEnsembleLogisticRegressionMetric(MaxEnsembleMetric):
    def create_metric(self) -> ObfMetric:
        return LogisticRegressionMetric(
            self.model,
            self.data_module,
            self.config,
            self.logger,
        )


class VAEMetric(ObfMetric):
    def __init__(
        self,
        model: ModelBase,
        data_module: ConceptDataModule,
        config: MetricConfig,
        logger: Logger,
        probe_path: Optional[str] = None,
        pca_path: Optional[str] = None,
    ):
        latent_dim: int = 512  # Size of VAE latent space

        self.latent_dim = latent_dim
        self.vaes: Dict[int, VAE] = {}

        # Learning hyperparams
        self.lr = 1e-4
        self.noise = True
        self.kld_weight = 1.0
        super().__init__(model, data_module, config, logger, probe_path, pca_path)

    def fit(self, reps_dataset: RepsDatasetT, **kwargs):
        # Extract negative examples (benign samples)
        negative_reps_list: List[Float[Tensor, "b layers seq_len hidden_dim"]] = [
            rep for rep, label in reps_dataset if label == 0
        ]

        _, layers, _, hidden_dim = negative_reps_list[0].shape
        # Concatenate negative reps into a single tensor
        negative_reps = torch.cat(
            [rep.view(-1, rep.shape[1], rep.shape[-1]) for rep in negative_reps_list],
            dim=0,
        )
        # Shuffle along axis 0 (examples dimension)
        shuffle_idx = torch.randperm(negative_reps.shape[0])
        negative_reps = negative_reps[shuffle_idx]

        device = self.device

        # Train a VAE for each layer
        for layer_idx, layer in enumerate(self.config.layers):
            layer_reps = negative_reps[:, layer_idx, :].to(device)

            # Initialize VAE
            vae = VAE(input_dim=hidden_dim, latent_dim=self.latent_dim, dtype=torch.float32).to(
                device
            )

            optimizer = torch.optim.Adam(vae.parameters(), lr=self.lr)
            # Train VAE
            batch_size = 64
            num_epochs = 2
            dataset = torch.utils.data.TensorDataset(layer_reps)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            print(f"Training VAE for layer {layer}")
            for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs", position=0, leave=True):
                total_loss = 0
                for batch in tqdm.tqdm(dataloader, desc="Batches", position=1, leave=False):
                    optimizer.zero_grad()

                    assert len(batch) == 1
                    x = batch[0].to(torch.float32)

                    assert x.ndim == 2, f"Expected 2D tensor, but got {x.ndim}D tensor"
                    assert x.shape[1] == hidden_dim

                    recon_x, mu, log_var = vae(x, noise=self.noise)
                    assert recon_x.shape == x.shape

                    loss = vae.loss_function(
                        reconstruction=recon_x,
                        input=x,
                        mu=mu,
                        log_var=log_var,
                        kld_weight=self.kld_weight,
                    )["loss"]
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    self.logger.log({f"vae_loss_layer_{layer}": loss.item()})

            vae.eval()
            self.vaes[layer_idx] = vae

    def evaluate(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        **kwargs,
    ) -> Float[Tensor, "b layers seq_len"]:
        b, n_layers, seq_len, hidden_dim = reps.shape
        device = self.device

        reps = reps.to(torch.float32)
        # Initialize output tensor
        scores = torch.zeros((b, n_layers, seq_len), device=device)

        for b_num, single_seq_rep in enumerate(reps):
            assert single_seq_rep.ndim == 3
            for layer_idx, layer in enumerate(self.config.layers):
                layer_reps = single_seq_rep[layer_idx, :, :].to(device)

                assert layer_reps.ndim == 2
                vae = self.vaes[layer_idx]

                # Calculate reconstruction error
                recon_x, mu, log_var = vae(layer_reps, noise=False)
                loss = vae.loss_function(
                    reconstruction=recon_x,
                    input=layer_reps,
                    mu=mu,
                    log_var=log_var,
                    kld_weight=self.kld_weight,
                    reduce=False,
                )["loss"]

                scores[b_num, layer_idx, :] = loss

        return scores

    def predict(
        self,
        reps: Float[Tensor, "b layers seq_len hidden_dim"],
        attention_mask: Optional[Bool[Tensor, "b seq_len"]] = None,
        layer_reduction: str = "mean",
        **kwargs,
    ) -> Float[Tensor, "b"]:
        b, n_layers, seq_len, hidden_dim = reps.shape
        scores: Float[Tensor, "b layers seq_len"] = self.evaluate(reps)

        if b > 1:
            assert attention_mask is not None, "Attention mask is required for batch evaluation"

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(~attention_mask, 0.0)
            scores = scores.sum(dim=2) / attention_mask.sum(dim=2).clamp(min=1)
        else:
            scores = scores.mean(dim=2)

        # Reduce over layers
        num_layers = len(self.config.layers)
        match layer_reduction:
            case "mean":
                scores = scores.sum(dim=1) / num_layers
            case "sum":
                scores = scores.sum(dim=1)
            case "max":
                scores = scores.max(dim=1).values
            case _:
                raise ValueError(f"Invalid layer_reduction: {layer_reduction}")

        assert scores.shape == (b,), f"Expected shape (b,), but got {scores.shape}"
        return scores

    def load_pretrained(
        self,
        path: Optional[str] = None,
        probe_state: Optional[Any] = None,
    ):
        assert (
            path is not None or probe_state is not None
        ), "Must provide either path or probe_state"
        if path is not None:
            with open(path, "rb") as f:
                self.vaes = pickle.load(f)
        else:
            self.vaes = probe_state

    def save_probe(
        self,
        path: Optional[str] = None,
    ) -> None | Any:
        if path is not None:
            with open(path, "wb") as f:
                pickle.dump(self.vaes, f)
            return None
        return self.vaes