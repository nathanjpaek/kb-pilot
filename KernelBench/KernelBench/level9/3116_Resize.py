import torch
import torch.nn as nn
import torch.nn.functional as F


class Resize(nn.Module):

    def __init__(self, input_size=[224, 224]):
        super(Resize, self).__init__()
        self.input_size = input_size

    def forward(self, input):
        x = F.interpolate(input, size=self.input_size, mode='bilinear',
            align_corners=True)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
