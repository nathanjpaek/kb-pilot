from _paritybench_helpers import _mock_config
import torch
import torch.utils.data
from torch import nn


class FusionMul(nn.Module):

    def __init__(self, input_channels, cfg):
        super(FusionMul, self).__init__()

    def forward(self, im_x, ra_x):
        x = torch.mul(im_x, ra_x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'cfg': _mock_config()}]
