import torch
from torch import nn
import torch.nn.functional as F


class Successor(nn.Module):
    """Successor successorEncoder model for ADDA."""

    def __init__(self):
        """Init Successor successorEncoder."""
        super(Successor, self).__init__()
        self.restored = False
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.dropout2 = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

    def forward(self, input):
        """Forward the Successor."""
        conv_out = F.relu(self.pool1(self.conv1(input)))
        conv_out = F.relu(self.pool2(self.dropout2(self.conv2(conv_out))))
        out = conv_out
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
