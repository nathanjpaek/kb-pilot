from torch.nn import Module
import torch
from torch.nn.modules.module import Module
import torch.optim
import torch.nn.modules.loss


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1.0 / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'r': 4, 't': 4}]
