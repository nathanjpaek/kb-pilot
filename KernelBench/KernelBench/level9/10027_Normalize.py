import torch
import torch.nn as nn


class Normalize(nn.Module):

    def forward(self, waveform: 'torch.Tensor') ->torch.Tensor:
        return (waveform - waveform.mean()) / waveform.std()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
