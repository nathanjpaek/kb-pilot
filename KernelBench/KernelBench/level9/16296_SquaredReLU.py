from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class SquaredReLU(Module):
    """
    ## Squared ReLU activation

    $$y = {\\max(x, 0)}^2$$

    Squared ReLU is used as the activation function in the
     [position wise feedforward module](../feed_forward.html).
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: 'torch.Tensor'):
        x = self.relu(x)
        return x * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
