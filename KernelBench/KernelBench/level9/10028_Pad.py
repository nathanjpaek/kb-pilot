import torch
import torch.nn as nn
import torch.nn.functional as F


class Pad(nn.Module):

    def __init__(self, value: 'float', size: 'int'):
        super().__init__()
        self.value = value
        self.size = size

    def forward(self, waveform: 'torch.Tensor') ->torch.Tensor:
        return F.pad(waveform, (0, self.size - max(waveform.shape)),
            'constant', self.value)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'value': 4, 'size': 4}]
