import torch
import torch.nn as nn


class Conv2d_depthwise_sep(nn.Module):

    def __init__(self, nin, nout):
        super(Conv2d_depthwise_sep, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1,
            groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nin': 4, 'nout': 4}]
