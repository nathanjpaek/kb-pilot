import torch
import torch.nn as nn


class FlowHead(nn.Module):
    """
    Applies two 2D convolutions over an input feature map
    to generate a flow tensor of shape N x 2 x H x W.

    Parameters
    ----------
    input_dim : int, default: 128
        Number of input dimensions.
    hidden_dim : int, default: 256
        Number of hidden dimensions.
    """

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape N x input_dim x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 2 x H x W
        """
        return self.conv2(self.relu(self.conv1(x)))


def get_inputs():
    return [torch.rand([4, 128, 64, 64])]


def get_init_inputs():
    return [[], {}]
