import torch
import torch.utils.data
import torch.nn as nn


class Noise(nn.Module):

    def __init__(self):
        super(Noise, self).__init__()

    def forward(self, input, train=False):
        input = input * 255.0
        if train:
            noise = torch.nn.init.uniform_(torch.zeros_like(input), -0.5, 0.5)
            output = input + noise
            output = torch.clamp(output, 0, 255.0)
        else:
            output = input.round() * 1.0
            output = torch.clamp(output, 0, 255.0)
        return output / 255.0


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
