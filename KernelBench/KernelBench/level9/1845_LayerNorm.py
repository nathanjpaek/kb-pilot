from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """ layer normalization """

    def __init__(self, cfg, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden_dim))
        self.beta = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(std + self.eps)
        return self.gamma * x + self.beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(hidden_dim=4)}]
