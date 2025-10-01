import torch
from torch import nn
from torch.nn import functional as F


class QueryModule(nn.Module):
    """
    A neural module that takes as input a feature map and an attention and produces a feature
    map as output.

    Extended Summary
    ----------------
    A :class:`QueryModule` takes a feature map and an attention mask as input. It attends to the
    feature map via an elementwise multiplication with the attention mask, then processes this
    attended feature map via a series of convolutions to extract relevant information.

    For example, a :class:`QueryModule` tasked with determining the color of objects would output a
    feature map encoding what color the attended object is. A module intended to count would output
    a feature map encoding the number of attended objects in the scene.

    Parameters
    ----------
    dim: int
        The number of channels of each convolutional filter.
    """

    def __init__(self, dim: 'int'):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64]), torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'dim': 4}]
