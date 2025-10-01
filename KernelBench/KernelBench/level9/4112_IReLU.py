import math
import torch


class IReLU(torch.nn.Module):
    __constants__ = ['negative_slope', 'positive_slope']
    negative_slope: 'float'
    positive_slope: 'float'

    def __init__(self, negative_slope=math.tan(math.pi / 8), positive_slope
        =math.tan(3 * math.pi / 8)):
        super(IReLU, self).__init__()
        self.negative_slope = negative_slope
        self.positive_slope = positive_slope

    def forward(self, x):
        return torch.clamp(x, min=0) * self.positive_slope + torch.clamp(x,
            max=0) * self.negative_slope

    def inv(self, y):
        return torch.clamp(y, min=0) / self.positive_slope + torch.clamp(y,
            max=0) / self.negative_slope


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
