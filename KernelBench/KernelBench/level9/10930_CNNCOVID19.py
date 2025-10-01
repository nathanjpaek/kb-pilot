from _paritybench_helpers import _mock_config
import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNCOVID19(nn.Module):

    def __init__(self, args):
        super(CNNCOVID19, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.fc1 = nn.Linear(in_features=8 * 31 * 31, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=3)

    def forward(self, l):
        l = self.conv1(l)
        l = F.relu(l)
        l = F.max_pool2d(l, kernel_size=2)
        l = l.reshape(-1, 8 * 31 * 31)
        l = self.fc1(l)
        l = self.out(l)
        return F.log_softmax(l, dim=1)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'args': _mock_config()}]
