import torch
from torch import nn


class GatedActivation(nn.Module):
    """Activation function which computes actiation_fn(f) * sigmoid(g).
  
  The f and g correspond to the top 1/2 and bottom 1/2 of the input channels.
  """

    def __init__(self, activation_fn=torch.tanh):
        """Initializes a new GatedActivation instance.

    Args:
      activation_fn: Activation to use for the top 1/2 input channels.
    """
        super().__init__()
        self._activation_fn = activation_fn

    def forward(self, x):
        _, c, _, _ = x.shape
        assert c % 2 == 0, 'x must have an even number of channels.'
        x, gate = x[:, :c // 2, :, :], x[:, c // 2:, :, :]
        return self._activation_fn(x) * torch.sigmoid(gate)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
