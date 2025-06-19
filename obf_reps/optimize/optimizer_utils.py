import torch
from torch.optim import Adam


class ConstrainedAdam(torch.optim.Adam):
    """A variant of Adam where some of the parameters are constrained to have unit norm.

    This is useful for learnign constrained soft prompts.
    """

    def __init__(self, params, constrained_params, **kwargs):
        super().__init__(params, **kwargs)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=-1, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=-1, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=-1, keepdim=True)
