import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """Linear model applying on the logits alpha * x + beta.

    By default, this model is initialized as an identity operation.

    See https://arxiv.org/abs/1905.13260 for an example usage.

    :param alpha: A learned scalar.
    :param beta: A learned scalar.
    """

    def __init__(self, alpha=1.0, beta=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, inputs):
        return self.alpha * inputs + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
