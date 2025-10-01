import torch
import torch.nn as nn


class ConvPlus(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, g=1, bias=True):
        super(ConvPlus, self).__init__()
        self.cv1 = nn.Conv2d(c1, c2, (k, 1), s, (k // 2, 0), groups=g, bias
            =bias)
        self.cv2 = nn.Conv2d(c1, c2, (1, k), s, (0, k // 2), groups=g, bias
            =bias)

    def forward(self, x):
        return self.cv1(x) + self.cv2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'c1': 4, 'c2': 4}]
