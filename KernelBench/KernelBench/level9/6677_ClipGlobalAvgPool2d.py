import torch
from torch import nn


class FastGlobalAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0),
                x.size(1), 1, 1)


class ClipGlobalAvgPool2d(nn.Module):

    def __init__(self):
        super().__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
