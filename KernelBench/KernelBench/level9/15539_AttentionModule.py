import torch
from torch import nn
from torch.nn import functional as F


class AttentionModule(nn.Module):
    """
    A neural module that takes a feature map and attention, attends to the features, and produces
    an attention.

    Extended Summary
    ----------------
    A :class:`AttentionModule` takes input features and an attention and produces an attention. It
    multiplicatively combines its input feature map and attention to attend to the relevant region
    of the feature map. It then processes the attended features via a series of convolutions and
    produces an attention mask highlighting the objects that possess the attribute the module is
    looking for.

    For example, an :class:`AttentionModule` may be tasked with finding cubes. Given an input
    attention of all ones, it will highlight all the cubes in the provided input features. Given
    an attention mask highlighting all the red objects, it will produce an attention mask
    highlighting all the red cubes.

    Parameters
    ----------
    dim: int
        The number of channels of each convolutional filter.
    """

    def __init__(self, dim: 'int'):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.dim = dim

    def forward(self, feats, attn):
        attended_feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(attended_feats))
        out = F.relu(self.conv2(out))
        out = torch.sigmoid(self.conv3(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64]), torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'dim': 4}]
