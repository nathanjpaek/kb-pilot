import torch
import torch.nn as nn


class LearnableSinusoidEncoding(nn.Module):
    """Layer converts scalar input to Sinusoid Encoding with learnt scaling."""

    def __init__(self, dim, max_timescale_init=10000):
        """Initialize layer.

        Args:
            dim: Dimensionality of the sinusoid encoding, should be dividable
                by 2.
            max_timescale_init: Maximum time scale used during initialization.
        """
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / max_timescale_init ** (torch.arange(0, dim, 2).
            float() / dim)
        self.inv_freq = nn.Parameter(inv_freq, requires_grad=True)

    def forward(self, x):
        sinusoid_inp = torch.matmul(x[..., None], self.inv_freq[None, :])
        emb = torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb.view(*emb.shape[:-2], -1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
