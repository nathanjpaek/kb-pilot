import torch
import torch.nn as nn


class Conv(nn.Module):

    def __init__(self, chn_in, chn_out, ker_sz=3):
        super().__init__()
        self.c = nn.Conv2d(chn_in, chn_out, ker_sz, padding=ker_sz // 2,
            padding_mode='circular', bias=False)
        self.a = nn.ReLU()

    def forward(self, x):
        x = self.c(x)
        x = self.a(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'chn_in': 4, 'chn_out': 4}]
