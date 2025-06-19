from typing import List

import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int64
from torch import Tensor, nn
from transformers.tokenization_utils import PreTrainedTokenizer

from obf_reps.models import GenReturn, ModelBase, ModelConfig
from obf_reps.models.hf import HFHardPrompted, HFSoftPrompted


class Gemma2bSoftPrompted(HFSoftPrompted):
    """A soft-prompted version of the Gemma 2B model.

    This class implements soft prompting for the Gemma 2B model, where learnable continuous vectors
    are appended to the input embeddings. These vectors can be optimized during training.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "Gemma2ForCausalLM" == self.model.__class__.__name__


class Gemma2bHardPrompted(HFHardPrompted):
    """A hard-prompted version of the Gemma 2B model.

    This class implements hard prompting for the Gemma 2B model, where learnable tokens appended to
    the input tokens. These tokens are optimized during training.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "Gemma2ForCausalLM" == self.model.__class__.__name__
