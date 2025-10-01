from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn


def gelu(x):
    """ gelu activation function """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWiseFeedForward(nn.Module):
    """ feedForward neural networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.feedforward_dim)
        self.fc2 = nn.Linear(cfg.feedforward_dim, cfg.hidden_dim)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(hidden_dim=4, feedforward_dim=4)}]
