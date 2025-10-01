import torch
import torch.nn as nn
import torch.nn.functional as F


class Bicubic(nn.Module):

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        bicubic_output = F.interpolate(inputs, scale_factor=self.
            scale_factor, mode='bicubic')
        return bicubic_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
