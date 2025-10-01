from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class AvgConsensus(nn.Module):

    def __init__(self, cfg):
        super(AvgConsensus, self).__init__()
        pass

    def forward(self, input, dim=0):
        assert isinstance(input, torch.Tensor)
        output = input.mean(dim=dim, keepdim=False)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config()}]
