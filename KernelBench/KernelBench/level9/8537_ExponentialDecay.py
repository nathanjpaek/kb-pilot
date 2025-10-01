import torch
from torch import Tensor
from torch import nn
from torch.jit import Final


class ExponentialUpdate(nn.Module):
    alpha: 'Final[int]'

    def __init__(self, alpha: 'float'):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: 'Tensor', state: 'Tensor') ->Tensor:
        return x * (1 - self.alpha) + state * self.alpha


class ExponentialDecay(nn.Module):

    def __init__(self, alpha: 'float'):
        super().__init__()
        self.update_rule = ExponentialUpdate(alpha)

    def forward(self, x: 'Tensor', state: 'Optional[Tensor]'=None):
        out = torch.empty_like(x)
        if state is None:
            state = x[0]
        for t in range(x.shape[0]):
            state = self.update_rule(x[t], state)
            out[t] = state
        return out, state


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4}]
