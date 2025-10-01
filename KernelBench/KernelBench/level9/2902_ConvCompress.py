import torch
from torch import nn


class ConvCompress(nn.Module):

    def __init__(self, d_model, ratio=4, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, ratio, stride=ratio, groups
            =groups)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
