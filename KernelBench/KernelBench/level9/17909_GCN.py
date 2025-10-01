from torch.nn import Module
import torch
import torch.utils.data
from torch.nn import Conv1d
from torch.nn import ReLU


class GCN(Module):

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = Conv1d(num_node, num_node, kernel_size=1, padding=0,
            stride=1, groups=1, bias=True)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv1d(num_state, num_state, kernel_size=1, padding=0,
            stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_state': 4, 'num_node': 4}]
