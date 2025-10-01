import torch
import torch.nn as nn
import torch.nn.functional as F


class normrelu(nn.Module):

    def __init__(self):
        super(normrelu, self).__init__()

    def forward(self, x):
        dim = 1
        x = F.relu(x) / torch.max(x, dim, keepdim=True)[0]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
