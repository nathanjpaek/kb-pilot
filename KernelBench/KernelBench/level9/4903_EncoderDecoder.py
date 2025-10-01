import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()

    def forward(self, x):
        _b, _c, h, w = x.shape
        x = F.adaptive_max_pool2d(x, (h // 2, w // 2))
        x = F.interpolate(x, size=(h, w), mode='bilinear')
        return torch.sigmoid(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
