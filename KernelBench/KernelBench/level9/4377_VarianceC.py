import torch
import torch.nn as nn


class VarianceC(nn.Module):

    def __init__(self):
        super(VarianceC, self).__init__()

    def forward(self, x):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        sub_x = x.sub(mean_x)
        x = torch.mean(torch.mul(sub_x, sub_x), dim=1, keepdim=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
