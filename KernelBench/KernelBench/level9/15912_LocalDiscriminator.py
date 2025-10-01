import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class LocalDiscriminator(nn.Module):
    """The local discriminator class.

    A network that analyses the relation between the
    output of the encoder y, and the feature map M.
    It is called "local" because it compares y with
    each one of the features in M. So if M is a [64, 6, 6]
    feature map, and y is a [32] vector, the comparison is
    done concatenating y along each one of the 6x6 features
    in M: 
    (i) [32] -> [64, 1, 1]; (ii) [32] -> [64, 1, 2]
    ... (xxxvi) [32] -> [64, 6, 6]. 
    This can be efficiently done expanding y to have same 
    dimensionality as M such that:
    [32] torch.expand -> [32, 6, 6]
    and then concatenate on the channel dimension:
    [32, 6, 6] torch.cat(axis=0) -> [64, 6, 6] = [96, 6, 6]
    The tensor is then feed to the local discriminator.
    """

    def __init__(self, y_size, M_channels):
        super().__init__()
        self.c0 = nn.Conv2d(y_size + M_channels, 256, kernel_size=1)
        self.c1 = nn.Conv2d(256, 256, kernel_size=1)
        self.c2 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


def get_inputs():
    return [torch.rand([4, 8, 64, 64])]


def get_init_inputs():
    return [[], {'y_size': 4, 'M_channels': 4}]
