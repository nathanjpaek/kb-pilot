import torch
import torch.nn as nn


class SubpixelConvolutionLayer(nn.Module):

    def __init__(self, channels: 'int'=64) ->None:
        """
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
        """
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride
            =1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
