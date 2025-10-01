import torch
from torch import nn


class FCUDown(nn.Module):

    def __init__(self, c1, c2, dw_stride):
        super().__init__()
        self.conv_project = nn.Conv2d(c1, c2, 1, 1, 0)
        self.sample_pooling = nn.AvgPool2d(dw_stride, dw_stride)
        self.ln = nn.LayerNorm(c2)
        self.act = nn.GELU()

    def forward(self, x, x_t):
        x = self.conv_project(x)
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'c1': 4, 'c2': 4, 'dw_stride': 1}]
