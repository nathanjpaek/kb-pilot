import torch
from torch import nn


class CondInjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, labels, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        labels = labels.view(-1, 1, 1, 1)
        batch, _, height, width = image.shape
        image.new_ones(batch, 1, height, width) / (labels + 1)
        return image + self.weight * noise


def get_inputs():
    return [torch.rand([256, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
