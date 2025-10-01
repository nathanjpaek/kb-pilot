import torch
import torch.nn as nn
import torch.nn.functional


class Cblock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(Cblock, self).__init__()
        self.block = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride,
            padding=1, bias=True)

    def forward(self, x):
        return self.block(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
