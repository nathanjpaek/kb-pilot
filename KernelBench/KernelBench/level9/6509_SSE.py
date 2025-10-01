import torch
import torch.nn as nn


class SSE(nn.Module):
    """SSE : Channel Squeeze and Spatial Excitation block.

    Paper : <https://arxiv.org/abs/1803.02579>

    Adapted from
    <https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178>
    """

    def __init__(self, in_channels):
        """Constructor method for SSE class.

        Args:
            in_channels : The number of input channels in the feature map.
        """
        super(SSE, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=1,
            kernel_size=1, stride=1)

    def forward(self, x) ->torch.Tensor:
        """Forward Method.

        Args:
            x: The input tensor of shape (batch, channels, height, width)

        Returns:
            Tensor of same shape
        """
        x_inp = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = torch.mul(x_inp, x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
