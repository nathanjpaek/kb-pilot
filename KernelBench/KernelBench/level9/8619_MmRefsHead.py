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


class MmRefsHead(nn.Module):

    def __init__(self, cfg):
        super(MmRefsHead, self).__init__()
        self.cfg = cfg
        self.dense = nn.Linear(cfg.HSIZE, cfg.HSIZE)
        self.layer_norm = LayerNorm(cfg.HSIZE, eps=1e-12)
        self.dense_rank = nn.Linear(cfg.HSIZE, 1)
        self.dense_reg = nn.Linear(cfg.HSIZE, 4)

    def forward(self, x):
        x = self.layer_norm(gelu(self.dense(x)))
        pred_rank = self.dense_rank(x).squeeze(-1)
        pred_reg = self.dense_reg(x)
        return pred_rank, pred_reg


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(HSIZE=4)}]
