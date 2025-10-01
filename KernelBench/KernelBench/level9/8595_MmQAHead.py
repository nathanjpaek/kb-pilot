from _paritybench_helpers import _mock_config
import math
import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MmQAHead(nn.Module):

    def __init__(self, cfg, ans_size):
        super(MmQAHead, self).__init__()
        self.cfg = cfg
        self.dense0 = nn.Linear(cfg.HSIZE, cfg.HSIZE * 2)
        self.dense1 = nn.Linear(cfg.HSIZE * 2, ans_size)
        self.layer_norm = LayerNorm(cfg.HSIZE * 2, eps=1e-12)

    def forward(self, x_pooled):
        pred = self.dense1(self.layer_norm(gelu(self.dense0(x_pooled))))
        return pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(HSIZE=4), 'ans_size': 4}]
