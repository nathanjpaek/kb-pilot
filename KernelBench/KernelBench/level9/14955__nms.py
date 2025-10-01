import torch
import torch.utils.data
import torch
import torch.nn as nn


class _nms(nn.Module):

    def __init__(self):
        super(_nms, self).__init__()
        kernel = 3
        pad = (kernel - 1) // 2
        self.maxpool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=pad)

    def forward(self, heat):
        hmax = self.maxpool(heat)
        keep = (hmax == heat).float()
        return heat * keep


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
