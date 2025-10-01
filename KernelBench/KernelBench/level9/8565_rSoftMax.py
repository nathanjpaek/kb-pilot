import torch
import torch.nn as nn
import torch.nn.functional as F


class rSoftMax(nn.Module):
    """
    (radix-majorize) softmax class

    input is cardinal-major shaped tensor.
    transpose to radix-major
    """

    def __init__(self, groups=1, radix=2):
        super(rSoftMax, self).__init__()
        self.groups = groups
        self.radix = radix

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.groups, self.radix, -1).transpose(1, 2)
        x = F.softmax(x, dim=1)
        x = x.view(B, -1, 1, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
