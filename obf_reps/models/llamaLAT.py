from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from obf_reps.models import ModelConfig
from obf_reps.models.hf import HFHardPrompted, HFSoftPrompted


class LlamaLATSoftPrompted(HFSoftPrompted):
    """A soft-prompted version of the Llama model with Latent Adversarial Training.

    This class implements soft prompting for the Llama model with LAT (latent adversarial
    training), where learnable continuous vectors are appended to the input embeddings. These
    vectors can be optimized during training. LAT is a technique that improves model robustness by
    training on adversarially perturbed latent representations.

    For more details on LAT, see:
    https://arxiv.org/abs/2403.05030
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "LAT" in self.model.config._name_or_path


class LlamaLATHardPrompted(HFHardPrompted):
    """A hard-prompted version of the Llama model with Latent Adversarial Training.

    This class implements hard prompting for the Llama model with LAT (latent adversarial
    training), where learnable tokens are appended to the input tokens. These tokens are optimized
    during training. LAT is a technique that improves model robustness by training on adversarially
    perturbed latent representations.

    For more details on LAT, see:
    https://arxiv.org/abs/2403.05030
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        super().__init__(model, tokenizer, config)
        # Check model correct
        assert "LAT" in self.model.config._name_or_path
