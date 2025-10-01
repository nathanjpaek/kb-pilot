import torch
import torch.nn as nn


class ImageProcessor(nn.Module):

    def __init__(self, init_image_embedding_size, embedding_size):
        super().__init__()
        self.conv = nn.Conv2d(init_image_embedding_size, embedding_size,
            kernel_size=1)

    def forward(self, image_encoding):
        x = self.conv(image_encoding)
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'init_image_embedding_size': 4, 'embedding_size': 4}]
