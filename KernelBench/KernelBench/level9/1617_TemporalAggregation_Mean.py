from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
from math import sqrt as sqrt
from itertools import product as product


class TemporalAggregation_Mean(nn.Module):

    def __init__(self, cfg):
        super(TemporalAggregation_Mean, self).__init__()
        self.K = cfg.K

    def forward(self, s):
        s = s.view(s.size(0) // self.K, self.K, s.size(1), s.size(2), s.size(3)
            )
        s = torch.mean(s, dim=1)
        return s


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(K=4)}]
