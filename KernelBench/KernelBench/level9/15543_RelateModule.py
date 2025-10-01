import torch
from torch import nn
from torch.nn import functional as F


class RelateModule(nn.Module):
    """
    A neural module that takes as input a feature map and an attention and produces an attention
    as output.

    Extended Summary
    ----------------
    A :class:`RelateModule` takes input features and an attention and produces an attention. It
    multiplicatively combines the attention and the features to attend to a relevant region, then
    uses a series of dilated convolutional filters to indicate a spatial relationship to the input
    attended region.

    Parameters
    ----------
    dim: int
        The number of channels of each convolutional filter.
    """

    def __init__(self, dim: 'int'):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, padding=8, dilation=8)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.conv6 = nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.conv6.weight)
        self.dim = dim

    def forward(self, feats, attn):
        feats = torch.mul(feats, attn.repeat(1, self.dim, 1, 1))
        out = F.relu(self.conv1(feats))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = torch.sigmoid(self.conv6(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64]), torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {'dim': 4}]
