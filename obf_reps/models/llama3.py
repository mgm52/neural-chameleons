from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from obf_reps.models import ModelBase, ModelConfig
from obf_reps.models.hf import HFHardPrompted, HFSoftPrompted


class Llama3SoftPrompted(HFSoftPrompted):
    """A soft-prompted version of the Llama 3 model.

    This class implements soft prompting for the Llama 3 model, where learnable continuous vectors
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
        assert "Llama" in self.model.__class__.__name__


class Llama3HardPrompted(HFHardPrompted):
    """A hard-prompted version of the Llama 3 model.

    This class implements hard prompting for the Llama 3 model, where learnable tokens appended to
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
        assert "Llama" in self.model.__class__.__name__
