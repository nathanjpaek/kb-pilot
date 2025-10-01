import torch
import torch.nn as nn
import torch.utils.data


class SubpixelConvolutionLayer(nn.Module):

    def __init__(self, channels: 'int') ->None:
        """
        Args:
            channels (int): Number of channels in the input image.
        """
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
