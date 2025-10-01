import torch
from torch import nn
import torch.nn.functional as F


class ClsHead(nn.Module):
    """
    Class orientation

    Args:

        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, class_dim, **kwargs):
        super(ClsHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, class_dim)

    def forward(self, x):
        x = self.pool(x)
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1]))
        x = self.fc(x)
        if not self.training:
            x = F.softmax(x, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'class_dim': 4}]
