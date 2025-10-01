import torch
import torch.nn as nn


def channel_norm(image):
    img = image.flatten(2)
    avg = img.mean(-1)[:, :, None, None]
    var = img.var(-1)[:, :, None, None]
    return (image - avg) / torch.sqrt(var + 1e-06)


class ChannelNorm(nn.Module):

    def forward(self, image):
        return channel_norm(image)

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
