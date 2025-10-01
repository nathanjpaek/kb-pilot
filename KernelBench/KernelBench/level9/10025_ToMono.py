import torch
import torch.nn as nn


class ToMono(nn.Module):

    def forward(self, waveform: 'torch.Tensor') ->torch.Tensor:
        return torch.mean(waveform, dim=0, keepdim=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
