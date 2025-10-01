import torch
import torch.nn as nn


class UpSampling(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.unpool1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv1d(in_c, in_c, 3, padding=1)
        self.unpool2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv1d(in_c, in_c, 3, padding=1)

    def forward(self, x, x2):
        out1 = self.conv1(self.unpool1(x.transpose(-1, -2))).transpose(-1, -2)
        if x2 is not None:
            out1 += x2
        out2 = self.conv2(self.unpool2(out1.transpose(-1, -2))).transpose(-
            1, -2)
        return out1, out2


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 8, 4])]


def get_init_inputs():
    return [[], {'in_c': 4}]
