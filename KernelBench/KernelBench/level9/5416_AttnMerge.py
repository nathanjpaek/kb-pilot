import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnMerge(nn.Module):

    def __init__(self, hn_size):
        super(AttnMerge, self).__init__()
        self.fc = nn.Linear(hn_size, hn_size, bias=False)

    def forward(self, x):
        hx = self.fc(x)
        alpha = F.softmax(hx, dim=1)
        out = torch.sum(alpha * x, dim=1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hn_size': 4}]
