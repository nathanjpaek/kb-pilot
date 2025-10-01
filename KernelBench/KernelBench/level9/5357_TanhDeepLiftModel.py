import torch
import torch.nn as nn


class TanhDeepLiftModel(nn.Module):
    """
    Same as the ReLUDeepLiftModel, but with activations
    that can have negative outputs
    """

    def __init__(self) ->None:
        super().__init__()
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

    def forward(self, x1, x2):
        return 2 * self.tanh1(x1) + 2 * self.tanh2(x2 - 1.5)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
