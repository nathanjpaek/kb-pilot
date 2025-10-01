import torch
from torch import nn


class SoftmaxImage(nn.Module):
    """Apply Softmax on an image.

    Softmax2d applies on second dimension (i.e. channels), which is
    not what I want. This applies along the H and W dimensions, where
    (N, C, H, W) is the size of the input.

    """

    def __init__(self, channels, height, width):
        super(SoftmaxImage, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = x.view(-1, self.channels, self.height * self.width)
        x = self.softmax(x)
        x = x.view(-1, self.channels, self.height, self.width)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'height': 4, 'width': 4}]
