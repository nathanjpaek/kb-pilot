import torch
from torch import nn


class PixelSort(nn.Module):
    """The inverse operation of PixelShuffle
    Reduces the spatial resolution, increasing the number of channels.
    Currently, scale 0.5 is supported only.
    Later, torch.nn.functional.pixel_sort may be implemented.
    Reference:
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/modules/pixelshuffle.html#PixelShuffle
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/functional.html#pixel_shuffle
    """

    def __init__(self, upscale_factor=0.5):
        super(PixelSort, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, 2, 2, h // 2, w // 2)
        x = x.permute(0, 1, 5, 3, 2, 4).contiguous()
        x = x.view(b, 4 * c, h // 2, w // 2)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
