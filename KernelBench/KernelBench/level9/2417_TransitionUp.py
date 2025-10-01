import torch
from torch import Tensor
import torch.nn as nn


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:xy2 + max_height, xy1:xy1 + max_width]


class TransitionUp(nn.Module):
    """
    Scale the resolution up by transposed convolution
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', scale: 'int'=2
        ):
        super().__init__()
        if scale == 2:
            self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=3, stride=2, padding
                =0, bias=True)
        elif scale == 4:
            self.convTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=
                in_channels, out_channels=out_channels, kernel_size=3,
                stride=2, padding=0, bias=False), nn.BatchNorm2d(
                out_channels), nn.ConvTranspose2d(in_channels=out_channels,
                out_channels=out_channels, kernel_size=3, stride=2, padding
                =0, bias=True))

    def forward(self, x: 'Tensor', skip: 'Tensor'):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
