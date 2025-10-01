import torch
import torch.nn as nn


def channel_scale(image):
    img = image.flatten(2)
    vmin = img.min(-1)[0][:, :, None, None]
    vmax = img.max(-1)[0][:, :, None, None]
    return (image - vmin) / (vmax - vmin + 1e-06)


class ChannelScale(nn.Module):

    def forward(self, image):
        return channel_scale(image)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
