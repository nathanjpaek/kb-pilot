import torch
import torch.nn as nn


class Denoise_NormalizeLayer(nn.Module):

    def __init__(self):
        super(Denoise_NormalizeLayer, self).__init__()

    def forward(self, inputs: 'torch.tensor'):
        permute_RGBtoBGR = [2, 1, 0]
        inputs = inputs[:, permute_RGBtoBGR, :, :]
        out = inputs / 0.5 - 1
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
