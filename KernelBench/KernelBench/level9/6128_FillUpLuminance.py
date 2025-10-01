import torch
import torch.nn


class FillUpLuminance(torch.nn.Module):

    def __init__(self):
        super(FillUpLuminance, self).__init__()

    def forward(self, color, luminance):
        return color + (1 - color) * luminance


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
