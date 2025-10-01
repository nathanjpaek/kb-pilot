import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """A gaussian noise module.

    Args:
        stddev (float): The standard deviation of the normal distribution.
                        Default: 0.1.

    Shape:
        - Input: (batch, *)
        - Output: (batch, *) (same shape as input)
    """

    def __init__(self, mean=0.0, stddev=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, self.stddev)
        return x + noise


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
