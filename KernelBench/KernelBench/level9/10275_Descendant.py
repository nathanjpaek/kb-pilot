import torch
from torch import nn
import torch.nn.functional as F


class Descendant(nn.Module):
    """Descendant descendantEncoder model for ADDA."""

    def __init__(self):
        """Init Descendant descendantEncoder."""
        super(Descendant, self).__init__()
        self.restored = False
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

    def forward(self, input):
        """Forward the Descendant."""
        conv_out = F.relu(self.pool1(self.conv1(input)))
        out = conv_out
        return out


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
