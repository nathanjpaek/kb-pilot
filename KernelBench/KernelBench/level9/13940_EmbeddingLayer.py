import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(self, in_channel, out_channel, img_size, patch_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding1 = nn.Conv2d(in_channel, out_channel, kernel_size=
            patch_size, stride=patch_size, padding=0)

    def forward(self, x):
        out = self.embedding1(x).flatten(2).transpose(1, 2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channel': 4, 'out_channel': 4, 'img_size': 4,
        'patch_size': 4}]
