import torch
from typing import Optional
import torch.nn as nn


class NoiseInjection(nn.Module):
    """
    Model injects noisy bias to input tensor
    """

    def __init__(self) ->None:
        """
        Constructor method
        """
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.float32),
            requires_grad=True)

    def forward(self, input: 'torch.Tensor', noise:
        'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor
        :param noise: (Optional[torch.Tensor]) Noise tensor
        :return: (torch.Tensor) Output tensor
        """
        if noise is None:
            noise = torch.randn(input.shape[0], 1, input.shape[2], input.
                shape[3], device=input.device, dtype=torch.float32)
        return input + self.weight * noise


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
