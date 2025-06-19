from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from obf_reps.models import ModelConfig
from obf_reps.models.hf import HFHardPrompted, HFSoftPrompted


class LlamaRRSoftPrompted(HFSoftPrompted):
    """A soft-prompted version of the Llama model with Representation Rerouting.

    This class implements soft prompting for the Llama model with RR (representation rerouting),
    where learnable continuous vectors are appended to the input embeddings. These vectors can be
    optimized during training. RR, also known as circuit breakers, is a technique that modifies the
    model's internal representations to avoid harmful behavior.

    For more details on RR, see:
    https://arxiv.org/abs/2406.04313v4
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "RR" in self.model.config._name_or_path


class LlamaRRHardPrompted(HFHardPrompted):
    """A hard-prompted version of the Llama model with Representation Rerouting.

    This class implements hard prompting for the Llama model with RR (representation rerouting),
    where learnable tokens are appended to the input tokens. These tokens are optimized during
    training. RR, also known as circuit breakers, is a technique that modifies the model's internal
    representations to avoid harmful behavior.

    For more details on RR, see:
    https://arxiv.org/abs/2406.04313v4
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "RR" in self.model.config._name_or_path
