import torch
from torch import nn
from torch.nn import functional as F


class ContractingBlock(nn.Module):
    """
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3_0 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv3x3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        """
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        """
        fx = F.relu(self.conv3x3_0(x))
        fx = F.relu(self.conv3x3_1(fx))
        return fx


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
