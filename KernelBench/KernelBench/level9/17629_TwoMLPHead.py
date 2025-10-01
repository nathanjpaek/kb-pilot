import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, out_channels=1024):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, out_channels)
        self.fc7 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
