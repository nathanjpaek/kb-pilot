import torch
import torch.nn.functional as F
import torch.nn
import torch.nn as nn


class SE_Connect(nn.Module):

    def __init__(self, channels, s=4):
        super().__init__()
        assert channels % s == 0, '{} % {} != 0'.format(channesl, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
