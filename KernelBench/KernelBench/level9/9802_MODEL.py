from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class MODEL(nn.Module):

    def __init__(self, args):
        super(MODEL, self).__init__()
        self.fc = nn.Linear(args.in_dim, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.sigmoid(self.fc(x)).flatten()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(in_dim=4)}]
