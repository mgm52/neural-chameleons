import copy
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from jaxtyping import Float
from torch import Tensor
from torch.optim.adam import Adam
from tqdm.autonotebook import tqdm
from transformers import set_seed
from transformers.cache_utils import HybridCache

from obf_reps.logging import Logger
from obf_reps.models import ModelBase
from obf_reps.optimize.flrt_utils import AttackBuffer
from obf_reps.optimize.gcg_utils import (
    check_refusal_completions,
    get_nonascii_toks,
    sample_control,
)
from obf_reps.optimize.loss import LossFunctionBase
from obf_reps.optimize.optimizer_utils import ConstrainedAdam


@dataclass
class OptimizerConfig:
    """Configuration for optimizer initialization.

    This class defines the configuration parameters used to initialize and control
    different optimizers during training.

    Attributes:
        lr: Learning rate for optimization
        num_steps: Number of optimization steps to run in INNNER LOOP of optimizers. Only the discrete
            optimizers use this.
        optim_str_init: Initial string for optimization
        search_width: Width of search space for GCG optimizer
        topk: Number of top candidates to consider in GCG
        n_replace: Number of tokens to replace in GCG
        buffer_size: Size of buffer for storing candidates
        use_prefix_cache: Whether to cache prefix computations
        filter_ids: Whether to filter invalid token IDs
        seed: Random seed for reproducibility
        allow_non_ascii: Whether to allow non-ASCII tokens
        eval_steps: Number of steps between evaluations
        eval_with_check_refusal: Whether to check for refusal during eval
        check_refusal_min_loss: Minimum loss threshold for refusal check
        early_stopping: Whether to use early stopping
        early_stopping_min_loss: Loss threshold for early stopping
        k1: FLRT-specific parameter for candidate selection
        k2: FLRT-specific parameter for candidate generation
        p_add: FLRT probability of adding tokens
        p_swap: FLRT probability of swapping tokens
        p_del: FLRT probability of deleting tokens
        init_len: Initial length for FLRT optimization
        generator_weight: Weight for generator loss in FLRT
        monitor_weight: Weight for monitor loss in FLRT
    """

    lr: float
    num_steps: int = 500
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

    # GCG-specific
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_prefix_cache: bool = True
    filter_ids: bool = True
    seed: int | None = None
    allow_non_ascii: bool = False
    eval_steps: int = 10
    eval_with_check_refusal: bool = False
    check_refusal_min_loss: float = 0.1
    early_stopping: bool = False
    early_stopping_min_loss: float = 0.1

    # FLRT-specific
    k1: int = 8
    k2: int = 15
    p_add: float = 0.5
    p_swap: float = 0.25
    p_del: float = 0.25
    init_len: int = 10
    generator_weight: float = 1.0
    monitor_weight: float = 0.0


class OptimizerBase(ABC):
    """Abstract base class for optimizers.

    This class defines the interface for optimizers that update model learnable parameters
    during training. Optimizers take a model, loss function, and logger and implement
    the optimization logic in their step() method.

    Args:
        model: Model to optimize
        loss_fn: Loss function to minimize
        logger: Logger for tracking metrics
        config: Optimizer configuration
    """

    def __init__(
        self,
        model: ModelBase,
        loss_fn: LossFunctionBase,
        logger: Logger,
        config: OptimizerConfig,
    ):
        self.loss_fn = loss_fn
        self.logger = logger
        self.config = config

        # Model stuff
        self.model = model
        self.tokenizer = model.tokenizer
        self.embedding_layer = model.model.get_input_embeddings()

        # Set seed
        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

    @abstractmethod
    def step(
        self,
        batch: Tuple[List[str], List[str], List[str]],
    ) -> None:
        """Given a batch of data, updates the model's tunable params."""
        ...


class ContinuousGradientOptimizer(OptimizerBase, ABC):
    """Uses gradients to directly update tunable parameters.

    This is used for tuning soft prompts.
    """

    @abstractmethod
    def init_optimizer(self, model: ModelBase, lr: float) -> torch.optim.Optimizer:
        """Initialize the optimizer with the tunable parameters from the model."""
        ...

    def __init__(
        self,
        model: ModelBase,
        loss_fn: LossFunctionBase,
        logger: Logger,
        config: OptimizerConfig,
    ):
        super().__init__(model, loss_fn, logger, config)

        self.optimizer = self.init_optimizer(model, config.lr)

    def step(self, batch) -> None:

        input_, behavior_target, rep_source = batch

        # Forward pass for behavior loss term
        behavior_output = self.model.forward_from_string(
            input_text=input_, target_text=behavior_target
        )

        # Forward pass for representation loss term
        rep_output = self.model.forward_from_string(input_text=input_, target_text=rep_source)

        behavior_target_input_ids, _ = self.model.tokenize(
            behavior_target, add_chat_template=False, add_special_tokens=False
        )

        # Compute loss
        loss: Float[Tensor, "b"] = self.loss_fn.compute_loss(
            behavior_logits=behavior_output.target_logits,
            behavior_target=behavior_target_input_ids,
            input_reps=rep_output.input_reps,
            target_reps=rep_output.target_reps,
            behavior_loss_mask=behavior_output.loss_mask,
            target_rep_loss_mask=rep_output.loss_mask,
        )
        loss = loss.mean().squeeze()

        self.logger.log({"loss": loss.item()})

        loss.backward()

        # Log softprompt and grad norms
        softprompt_norm = torch.norm(self.model.tunable_params.params).item()
        self.logger.log({"softprompt_norm": softprompt_norm})

        grad_norm = torch.norm(self.model.tunable_params.params.grad).item()

        with torch.no_grad():
            # Clip the gradient norm to 1
            torch.nn.utils.clip_grad_norm_(self.model.tunable_params.params, max_norm=1.0)
            # Recalculate the clipped grad norm for logging
            grad_norm = torch.norm(self.model.tunable_params.params.grad).item()

        self.logger.log({"grad_norm": grad_norm})

        # Take optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()


class AdamOptimizer(ContinuousGradientOptimizer):
    """Uses Adam to update tunable parameters.

    This is used for tuning soft prompts.
    """

    def init_optimizer(self, model: ModelBase, lr: float) -> torch.optim.Optimizer:
        return Adam([model.tunable_params.params], lr=lr, eps=1e-5)


class ConstrainedAdamOptimizer(ContinuousGradientOptimizer):
    """Uses Constrained Adam to update tunable parameters.

    This is used for learning constrained soft prompts.
    """

    def init_optimizer(self, model: ModelBase, lr: float) -> torch.optim.Optimizer:
        return ConstrainedAdam(
            [model.tunable_params.params], [model.tunable_params.params], lr=lr, eps=1e-5
        )


class GCGOptimizer(OptimizerBase):
    """Uses GCG to update hard prompt tunable parameters. Note this optimizer ONLY works for hard
    prompts.

    An implimentation of the GCG method from https://arxiv.org/abs/2307.15043

    note: this optimizer is designed to be run with a single batch
        and num steps (in optimizer config) to be set to large values.
        each call to step starts the optimization from scratch.
    """

    use_grad = True  # type: ignore

    def __init__(
        self,
        model: ModelBase,
        loss_fn: LossFunctionBase,
        logger: Logger,
        config: OptimizerConfig,
    ):
        super().__init__(model=model, loss_fn=loss_fn, logger=logger, config=config)
        self.model = model
        self.tokenizer = model.tokenizer
        self.num_steps = config.num_steps
        self.logger = logger

        self.optim_str_init = config.optim_str_init
        self.allow_non_ascii = config.allow_non_ascii
        self.search_width = config.search_width
        self.use_prefix_cache = config.use_prefix_cache
        if (
            config.use_prefix_cache
            and self.model.model.config.use_cache != config.use_prefix_cache
        ):
            print(f"WARNING: setting model.config.use_cache={config.use_prefix_cache}")
            self.model.model.config.use_cache = config.use_prefix_cache

        # Eval Vars
        self.eval_steps = config.eval_steps
        self.eval_with_check_refusal = config.eval_with_check_refusal
        self.check_refusal_min_loss = config.check_refusal_min_loss
        self.early_stopping = config.early_stopping
        self.early_stopping_min_loss = config.early_stopping_min_loss
        self.starting_search_batch_size = self.search_width

    def step(self, batch) -> None:
        """Main step function for the GCG optimizer."""

        self.search_batch_size = (
            self.starting_search_batch_size
            if self.starting_search_batch_size
            else self.search_width
        )

        input, behavior_target, rep_source = batch

        input = input[0]
        behavior_target = behavior_target[0]
        rep_source = rep_source[0]
        print("Using input:", input)
        print("Using behavior_target:", behavior_target)

        model = self.model
        device = model.model.device
        tokenizer = self.tokenizer

        num_steps = self.num_steps
        adv_string_init = self.optim_str_init
        allow_non_ascii = self.allow_non_ascii
        search_width = self.search_width

        eval_steps = self.eval_steps
        eval_with_check_refusal = self.eval_with_check_refusal
        check_refusal_min_loss = self.check_refusal_min_loss
        early_stopping = self.early_stopping
        early_stopping_min_loss = self.early_stopping_min_loss

        embed_layer = model.model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[
            0
        ]  # can be larger than tokenizer.vocab_size for some models
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(device))

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)
        # remove redundant tokens
        not_allowed_tokens = torch.unique(not_allowed_tokens)

        def get_ids_and_mask(input_str, **kwargs):
            return (
                tokenizer(input_str, return_tensors="pt", **kwargs)["input_ids"].to(device),
                tokenizer(input_str, return_tensors="pt", **kwargs)["attention_mask"].to(device),
            )

        optim_ids, optim_attn_mask = get_ids_and_mask(adv_string_init, add_special_tokens=False)
        num_optim_tokens = len(optim_ids[0])

        messages = [
            {"role": "user", "content": input + "{optim_str}"},
        ]
        template = tokenizer.apply_chat_template(messages, tokenize=False)
        before_tc, after_tc = template.split("{optim_str}")

        before_ids, before_attn_mask = get_ids_and_mask(
            before_tc, padding=False, add_special_tokens=False
        )
        after_ids, after_attn_mask = get_ids_and_mask(
            after_tc, padding=False, add_special_tokens=False
        )
        target_ids, target_attn_mask = get_ids_and_mask(
            behavior_target, padding=False, add_special_tokens=False
        )
        before_embeds, after_embeds, target_embeds = (
            embed_layer(input_ids) for input_ids in (before_ids, after_ids, target_ids)
        )

        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            input_embeds = torch.cat([before_embeds], dim=1)
            with torch.no_grad():
                outputs = model.model(inputs_embeds=input_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values

        # ========== run optimization ========== #
        all_losses = []
        all_test_cases = []
        for i in tqdm(range(num_steps)):
            # ========== compute coordinate token_gradient ========== #
            # create input
            optim_ids_onehot = torch.zeros(
                (1, num_optim_tokens, vocab_size), device=device, dtype=model.model.dtype
            )
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(optim_ids_onehot.squeeze(0), vocab_embeds).unsqueeze(0)

            # forward pass
            if self.use_prefix_cache:
                input_embeds = torch.cat([optim_embeds, after_embeds], dim=1)
                input_attn_mask = torch.cat([optim_attn_mask, after_attn_mask], dim=1)
            else:
                input_embeds = torch.cat(
                    [before_embeds, optim_embeds, after_embeds],
                    dim=1,
                )
                input_attn_mask = torch.cat(
                    [before_attn_mask, optim_attn_mask, after_attn_mask],
                    dim=1,
                )

            behavior_output = self.model.forward_from_embeds(
                input_embeds=input_embeds,
                input_attn_mask=input_attn_mask,
                target_ids=target_ids,
                target_attn_mask=target_attn_mask,
            )
            loss: Float[Tensor, "b"] = self.loss_fn.compute_loss(
                behavior_logits=behavior_output.target_logits,
                behavior_target=target_ids,
                input_reps=behavior_output.input_reps,
                target_reps=behavior_output.target_reps,  # should be rep_output.target_reps,
                behavior_loss_mask=behavior_output.loss_mask,
                target_rep_loss_mask=behavior_output.loss_mask,  # should be rep_output.loss_mask,
            )
            loss = loss.mean().squeeze()

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices = sample_control(
                optim_ids.squeeze(0),
                token_grad.squeeze(0),
                search_width,
                topk=256,
                temp=1,
                not_allowed_tokens=not_allowed_tokens,
            )

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                # tokenize again
                tmp = tokenizer(
                    sampled_top_indices_text[j], return_tensors="pt", add_special_tokens=False
                ).to(device)["input_ids"][0]
                # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                if not torch.equal(tmp, sampled_top_indices[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])

            if len(new_sampled_top_indices) == 0:
                print("All removals; defaulting to keeping all")
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)

            if count >= search_width // 2:
                print("\nLots of removals:", count)

            new_search_width = search_width - count

            # ========== Compute loss on these candidates and take the argmin. ========== #
            # create input
            sampled_top_embeds = embed_layer(sampled_top_indices)
            if self.use_prefix_cache:
                input_embeds = torch.cat(
                    [
                        sampled_top_embeds,
                        after_embeds.repeat(new_search_width, 1, 1),
                    ],
                    dim=1,
                )
                input_attn_mask = torch.cat(
                    [
                        torch.ones(new_search_width, sampled_top_embeds.shape[1], device=device),
                        after_attn_mask.repeat(new_search_width, 1),
                    ],
                    dim=1,
                )
            else:
                input_embeds = torch.cat(
                    [
                        before_embeds.repeat(new_search_width, 1, 1),
                        sampled_top_embeds,
                        after_embeds.repeat(new_search_width, 1, 1),
                    ],
                    dim=1,
                )
                input_attn_mask = torch.cat(
                    [
                        before_attn_mask.repeat(new_search_width, 1),
                        torch.ones(new_search_width, sampled_top_embeds.shape[1], device=device),
                        after_attn_mask.repeat(new_search_width, 1),
                    ],
                    dim=1,
                )

            # Auto Find Batch Size for foward candidates (each time go OOM will decay search_batch_size // 2)
            loss = find_executable_batch_size(
                self.compute_candidates_loss, self.search_batch_size
            )(input_embeds, input_attn_mask, target_ids, target_attn_mask)

            # ========== Update the optim_ids with the best candidate ========== #
            optim_ids = sampled_top_indices[loss.argmin()].unsqueeze(0)
            self.model.tunable_params.params = F.one_hot(optim_ids, embed_layer.num_embeddings).to(
                dtype=embed_layer.weight.dtype, device=embed_layer.weight.device
            )

            test_case_ids = torch.cat([before_ids, optim_ids], dim=1)
            test_case = tokenizer.decode(test_case_ids[0])
            all_test_cases.append(test_case)
            current_loss = loss.min().item()
            self.logger.log({"loss": current_loss})
            all_losses.append(current_loss)

            # ========== Eval and Early Stopping ========== #
            verbose = True
            if (i % eval_steps == 0) or (i == num_steps - 1):
                p_output = f"\n===>Step {i}\n===>Test Case: {test_case}\n===>Loss: {current_loss}"
                if eval_with_check_refusal and current_loss < check_refusal_min_loss:
                    message = [
                        {"role": "user", "content": test_case},
                    ]
                    input_str = tokenizer.apply_chat_template(message, tokenize=False)
                    is_refusal, completions, _ = check_refusal_completions(
                        model.model, tokenizer, inputs=[input_str]
                    )
                    p_output += f"\n\n===>Completion: {completions[0]}"
                    # if not is_refusal[0]:
                    #     break

                if verbose:
                    print(p_output)

            del input_embeds, sampled_top_embeds, loss
            torch.cuda.empty_cache()
            gc.collect()

        logs = {
            "final_loss": current_loss,
            "all_losses": all_losses,
            "all_test_cases": all_test_cases,
        }

        return test_case, logs

    def compute_candidates_loss(
        self, search_batch_size, input_embeds, input_attn_mask, target_ids, target_attn_mask
    ):
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        all_loss = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]
                input_attn_mask_batch = input_attn_mask[i : i + search_batch_size]
                target_ids_batch = target_ids.repeat(input_embeds_batch.shape[0], 1)
                target_attn_mask_batch = target_attn_mask.repeat(input_embeds_batch.shape[0], 1)

                if self.use_prefix_cache:
                    # Expand prefix cache to match batch size
                    prefix_cache_batch = self.prefix_cache
                    current_batch_size = input_embeds_batch.shape[0]
                    prefix_cache_batch = []
                    for i in range(len(self.prefix_cache)):
                        prefix_cache_batch.append([])
                        for j in range(len(self.prefix_cache[i])):
                            prefix_cache_batch[i].append(
                                self.prefix_cache[i][j].expand(current_batch_size, -1, -1, -1)
                            )

            outputs = self.model.forward_from_embeds(
                input_embeds=input_embeds_batch,
                input_attn_mask=input_attn_mask_batch,
                target_ids=target_ids_batch,
                target_attn_mask=target_attn_mask_batch,
            )

            loss: Float[Tensor, "b"] = self.loss_fn.compute_loss(
                behavior_logits=outputs.target_logits,
                behavior_target=target_ids_batch,
                input_reps=outputs.input_reps,
                target_reps=outputs.target_reps,
                behavior_loss_mask=outputs.loss_mask,
                target_rep_loss_mask=outputs.loss_mask,
            )
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)

            del outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
        return torch.cat(all_loss, dim=0)


class FLRTOptimizer(OptimizerBase):
    """Optimizer for hard prompt tunable parameters. Note this optimizer ONLY works for hard
    prompts.

    An implimentation of the FLRT optimizer from https://arxiv.org/abs/2407.17447 with the slight
    change that we replace the worst candidate from the buffer, and don't use the additional
    loss functions outlined in the FLRT paper.

    NOTE: This implimentation assumes model.model is a standard huggingface model. This can
        lead to some errors if not the case (we interact with model.model a couple of times).
    NOTE: This optimizer is designed to be run with a single batch
        and num steps (in optimizer config) to be set to large values.
        Each call to step starts the optimization from scratch.

    This is the optimizer we use for all the hard prompt experiments in the paper.
    """

    def __init__(
        self,
        model: ModelBase,
        loss_fn: LossFunctionBase,
        logger: Logger,
        config: OptimizerConfig,
    ):
        super().__init__(model=model, loss_fn=loss_fn, logger=logger, config=config)

        self.model = model
        self.tokenizer = model.tokenizer
        self.num_steps = config.num_steps
        self.logger = logger

        self.config = config
        self.monitor_weight = config.monitor_weight
        self.generator_weight = config.generator_weight

    def step(self, batch):
        """Main step function for the FLRT optimizer.

        Note that this optimizer is designed to be run with a single batch and num steps (in
        optimizer config) to be set to large values. Each call to step starts the optimization from
        scratch.
        """

        input, behavior_target, rep_source = batch

        input = input[0]
        behavior_target = behavior_target[0]
        rep_source = rep_source[0]
        print("Using input:", input)
        print("Using behavior_target:", behavior_target)

        model = self.model

        messages = [
            {"role": "user", "content": input + "{optim_str}"},
        ]
        before_ids, after_ids, target_ids = self.get_ids(messages, behavior_target)
        print("before ids: ", model.tokenizer.batch_decode(before_ids))
        print("after ids: ", model.tokenizer.batch_decode(after_ids))
        print("target ids: ", model.tokenizer.batch_decode(target_ids))

        # Embed everything that doesn't get optimized
        embedding_layer = model.model.get_input_embeddings()
        before_embeds, after_embeds, target_embeds = (
            embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
        )

        buffer = AttackBuffer(model, init_len=self.config.init_len, size=self.config.buffer_size)

        # Compute the KV Cache for tokens that appear before the optimized tokens
        with torch.no_grad():
            output = model.model(inputs_embeds=before_embeds, use_cache=True)
            kv_cache = output.past_key_values

        losses = []
        monitor_losses = []
        generator_losses = []
        early_stopping_condition = []
        optim_strings = []
        optim_idss = []
        skip_cond = False

        for i in tqdm(range(self.config.num_steps)):
            if skip_cond:
                # continue with last values for all returns
                losses.append(losses[-1])
                monitor_losses.append(monitor_losses[-1])
                generator_losses.append(generator_losses[-1])
                optim_strings.append(optim_strings[-1])
                optim_idss.append(optim_idss[-1])
                early_stopping_condition.append(early_stopping_condition[-1])
                continue

            best_ids = buffer.get_best().squeeze(0)

            rand = torch.rand(1, device=model.device).item()
            if rand < self.config.p_add or best_ids.shape[0] < 5:
                op = "add"
            elif rand < self.config.p_add + self.config.p_swap:
                op = "swap"
            else:
                op = "delete"

            print(f"Applying op: {op}")

            candidate_idxs = torch.randint(0, best_ids.shape[0], (self.config.k1,))

            if op == "delete":
                new_attack_ids_list = []
                for idx in candidate_idxs:
                    new_ids = torch.cat((best_ids[:idx], best_ids[idx + 1 :]), dim=0).unsqueeze(0)
                    new_attack_ids_list.append(new_ids)
                new_attack_ids = torch.cat(new_attack_ids_list, dim=0)
            else:
                input_embeds = embedding_layer(best_ids.unsqueeze(0))
                candidate_ids = self.sample_candidates(
                    candidate_idxs,
                    self.config.k2,
                    input_embeds,
                    kv_cache,
                    before_embeds,
                )
                if op == "swap":
                    new_attack_ids_list = []
                    print(f"candidate_ids: {candidate_ids}, shape: {candidate_ids.shape}")
                    for idx in tqdm(range(candidate_ids.shape[0])):
                        swap_idx = candidate_idxs[idx]
                        new_ids = best_ids.clone()
                        new_ids[swap_idx] = candidate_ids[idx]
                        new_attack_ids_list.append(new_ids.unsqueeze(0))
                    new_attack_ids = torch.cat(new_attack_ids_list, dim=0)
                elif op == "add":
                    new_attack_ids_list = []
                    for idx in range(candidate_ids.shape[0]):
                        add_idx = candidate_idxs[idx]
                        new_ids = torch.cat(
                            [
                                best_ids[: add_idx + 1],
                                candidate_ids[idx : idx + 1],
                                best_ids[add_idx + 1 :],
                            ],
                            dim=0,
                        ).unsqueeze(0)
                        new_attack_ids_list.append(new_ids)
                    new_attack_ids = torch.cat(new_attack_ids_list, dim=0)

            new_ids = self.fixed_point_ids(new_attack_ids)
            input_embeds = torch.cat(
                [
                    embedding_layer(new_ids),
                    after_embeds.repeat(new_ids.shape[0], 1, 1),
                    target_embeds.repeat(new_ids.shape[0], 1, 1),
                ],
                dim=1,
            )

            loss, monitor_loss, generator_loss, forcing_condition = find_executable_batch_size(
                self.compute_candidates_loss, new_ids.shape[0]
            )(
                kv_cache=kv_cache,
                input_embeds=input_embeds,
                target_ids=target_ids,
                generator_weight=self.config.generator_weight,
                monitor_weight=self.config.monitor_weight,
                sampled_ids=new_ids,
                before_embeds=before_embeds,
            )

            # Take the bottom k1 from loss and update the buffer. Ignore the old buffer losses
            sorted_indices = loss.argsort()
            optim_id = new_ids[sorted_indices[0]]
            optim_str = model.tokenizer.decode(optim_id)
            optim_loss = loss[sorted_indices[0]]
            optim_monitor_loss = monitor_loss[sorted_indices[0]]
            optim_generator_loss = generator_loss[sorted_indices[0]]
            forcing_condition = forcing_condition[sorted_indices[0]]

            buffer.replace_worst(optim_id, optim_loss, optim_monitor_loss, optim_generator_loss)

            losses.append(optim_loss.item())
            monitor_losses.append(optim_monitor_loss.item())
            generator_losses.append(optim_generator_loss.item())
            optim_strings.append(optim_str)
            optim_idss.append(optim_id.tolist())

            self.logger.log({"buff_loss": buffer.losses.min().item()})
            opstr = optim_str.replace("\n", r"\n")
            print(f"Step {i}: Optim String = {opstr}")
            monitor_condition = optim_monitor_loss < torch.tensor(0.5, device=model.device).item()

        best_id: Float[Tensor, "n_optim_ids"] = buffer.get_best()
        best_params = (
            F.one_hot(optim_id, embedding_layer.num_embeddings)
            .to(dtype=embedding_layer.weight.dtype, device=embedding_layer.weight.device)
            .unsqueeze(0)
        )
        assert best_params.shape[0] == 1
        assert best_params.shape[2] == embedding_layer.num_embeddings
        self.model.tunable_params.params = best_params

        print("Best params are:")
        print(model.tokenizer.batch_decode(best_id))
        print()

        return {
            "losses": losses,
            "monitor_losses": monitor_losses,
            "generator_losses": generator_losses,
            "optim_strings": optim_strings,
            "optim_ids": optim_idss,
            "early_stopping": early_stopping_condition,
        }

    def sample_candidates(
        self,
        candidate_idxs: Tensor,
        k2: int,
        input_embeds: Tensor,
        kv_cache: tuple | None,
        before_embeds: Optional[Tensor] = None,
    ):
        with torch.no_grad():
            if kv_cache is not None and not isinstance(kv_cache, HybridCache):
                outputs = self.model.model(
                    inputs_embeds=input_embeds,
                    past_key_values=kv_cache,
                    output_hidden_states=False,
                )
                logits = outputs.logits
            else:
                assert (
                    before_embeds is not None
                ), "before_embeds must be provided if kv_cache is None"
                input_embeds = torch.cat([before_embeds, input_embeds], dim=1)
                outputs = self.model.model(inputs_embeds=input_embeds, output_hidden_states=False)
                logits = outputs.logits[..., before_embeds.shape[1] :, :]

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
        special_ids = [0, 1, 2]  # Hardcoded from tokenizer.all_special_ids for now
        probs[..., special_ids] = 0.0
        probs[..., self.tokenizer.vocab_size :] = 0.0
        sampled_ids = torch.multinomial(probs[candidate_idxs], num_samples=k2, replacement=False)
        selection = torch.randint(0, k2, (candidate_idxs.shape[0],))
        return sampled_ids[torch.arange(candidate_idxs.shape[0]), selection]

    def filter_ids(self, ids: Tensor) -> torch.Tensor:
        """Filters out sequeneces of token ids that change after retokenization.

        Args:
            ids : Tensor, shape = (search_width, n_optim_ids)
                token ids
            tokenizer : ~transformers.PreTrainedTokenizer
                the model's tokenizer

        Returns:
            filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
                all token ids that are the same after retokenization
        """
        ids_decoded = self.tokenizer.batch_decode(ids)
        filtered_ids = []

        for i in range(len(ids_decoded)):
            # Retokenize the decoded token ids
            ids_encoded = self.tokenizer(
                ids_decoded[i], return_tensors="pt", add_special_tokens=False
            ).to(ids.device)["input_ids"][0]
            if torch.equal(ids[i], ids_encoded):
                filtered_ids.append(ids[i])

        if len(filtered_ids) == 0:
            return []
        else:
            return torch.stack(filtered_ids)

    def fixed_point_ids(self, ids: Tensor) -> torch.Tensor:
        is_fixed_point = False
        ids_encoded = ids
        while not is_fixed_point:
            is_fixed_point = True
            ids_decoded = self.tokenizer.batch_decode(ids_encoded)
            encoded_ids = []

            for i in range(len(ids_decoded)):
                # Retokenize the decoded token ids
                ids_recoded = self.tokenizer(
                    ids_decoded[i], return_tensors="pt", add_special_tokens=False
                ).to(ids.device)["input_ids"][0]
                if not encoded_ids or len(encoded_ids[0]) == len(ids_recoded):
                    encoded_ids.append(ids_recoded)
                    if not torch.equal(ids_encoded[i], ids_recoded):
                        is_fixed_point = False

            ids_encoded = torch.stack(encoded_ids)

        return ids_encoded

    def compute_candidates_loss(
        self,
        batch_size: int,
        kv_cache: tuple | None,
        input_embeds: Tensor,
        target_ids: Tensor,
        generator_weight: float,
        monitor_weight: float,
        sampled_ids: Tensor = None,
        before_embeds: Optional[Tensor] = None,
    ):

        all_loss = []
        all_generator_loss = []
        all_monitor_loss = []
        all_forcing_condition = []
        for i in range(0, input_embeds.shape[0], batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + batch_size]
                sampled_ids_batch = sampled_ids[i : i + batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if kv_cache is not None and not isinstance(kv_cache, HybridCache):
                    kv_cache_batch = []
                    for i in range(len(kv_cache)):
                        kv_cache_batch.append([])
                        for j in range(len(kv_cache[i])):
                            kv_cache_batch[i].append(
                                kv_cache[i][j].expand(current_batch_size, -1, -1, -1)
                            )

                    outputs = self.model.model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=kv_cache_batch,
                        output_hidden_states=True,
                    )
                    logits = outputs.logits
                    # hidden_states = outputs.hidden_states
                    reps: Float[Tensor, "b_size layers seq_len hidden_size"] = torch.stack(
                        outputs.hidden_states, dim=1
                    )

                else:
                    assert (
                        before_embeds is not None
                    ), "before_embeds must be provided if kv_cache is None"
                    input_embeds_batch = torch.cat(
                        [before_embeds.repeat(current_batch_size, 1, 1), input_embeds_batch], dim=1
                    )

                    # We don't have attention mask because we are only dealing
                    # with a single string in this case, so ther are no
                    # pad tokens
                    outputs = self.model.model(
                        inputs_embeds=input_embeds_batch,
                        output_hidden_states=True,
                    )

                    logits = outputs.logits[..., before_embeds.shape[1] :, :]
                    reps: Float[Tensor, "b_size layers seq_len hidden_size"] = torch.stack(
                        outputs.hidden_states, dim=1
                    )

                # Here we need to use our loss
                shift = input_embeds.shape[1] - target_ids.shape[1]
                shift_logits = logits[..., shift - 1 : -1, :].contiguous()
                shift_labels = target_ids.repeat(current_batch_size, 1)
                model_preds = shift_logits.argmax(dim=-1)
                forcing_condition = torch.all(torch.eq(model_preds, shift_labels), dim=-1)

                input_reps = reps[:, :, : -target_ids.shape[1], :]
                target_reps = reps[:, :, -target_ids.shape[1] :, :]
                output_loss_mask = (shift_labels != -100).bool()

                loss: Float[Tensor, "b_size"] = self.loss_fn.compute_loss(
                    behavior_logits=shift_logits,
                    behavior_target=shift_labels,
                    input_reps=input_reps,
                    target_reps=target_reps,
                    behavior_loss_mask=output_loss_mask,
                    target_rep_loss_mask=output_loss_mask,
                )

                all_loss.append(loss)
                all_generator_loss.append(loss)
                all_monitor_loss.append(loss)
                all_forcing_condition.append(forcing_condition)

                del outputs
                self.clear_gpus()

        return (
            torch.cat(all_loss),
            torch.cat(all_monitor_loss),
            torch.cat(all_generator_loss),
            torch.cat(all_forcing_condition),
        )

    def clear_gpus(
        self,
    ):
        gc.collect()
        torch.cuda.empty_cache()

    def get_ids(
        self,
        messages: Union[str, List[dict]],
        target: str,
    ):
        template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if self.tokenizer.bos_token and template.startswith(self.tokenizer.bos_token):
            template = template.replace(self.tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")
        print(
            f"before, after, target : {before_str}\n\n_______\n\n{after_str}\n\n_______\n\n{target}"
        )

        # Tokenize everything that doesn't get optimized
        before_ids = self.tokenizer([before_str])["input_ids"]  # add BOS token
        after_ids = self.tokenizer([after_str], add_special_tokens=False)[
            "input_ids"
        ]  # don't add BOS token
        target_ids = self.tokenizer([target], add_special_tokens=False)[
            "input_ids"
        ]  # don't add BOS token
        before_ids, after_ids, target_ids = (
            torch.tensor(ids, device=self.model.device)
            for ids in (before_ids, after_ids, target_ids)
        )

        return before_ids, after_ids, target_ids
