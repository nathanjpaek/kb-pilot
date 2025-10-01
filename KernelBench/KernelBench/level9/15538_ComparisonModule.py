import torch
from torch import nn
from torch.nn import functional as F


class ComparisonModule(nn.Module):
    """
    A neural module that takes as input two feature maps and produces a feature map as output.

    Extended Summary
    ----------------
    A :class:`ComparisonModule` takes two feature maps as input and concatenates these. It then
    processes the concatenated features and produces a feature map encoding whether the two input
    feature maps encode the same property.

    This block is useful in making integer comparisons, for example to answer the question, "Are
    there more red things than small spheres?" It can also be used to determine whether some
    relationship holds of two objects (e.g. they are the same shape, size, color, or material).

    Parameters
    ----------
    dim: int
        The number of channels of each convolutional filter.
    """

    def __init__(self, dim: 'int'):
        super().__init__()
        self.projection = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, in1, in2):
        out = torch.cat([in1, in2], 1)
        out = F.relu(self.projection(out))
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
