import torch
from torch import nn


class FourierEmbedding(nn.Module):

    def __init__(self, features, height, width, **kwargs):
        super().__init__(**kwargs)
        self.projector = nn.Linear(2, features)
        self._height = height
        self._width = width

    def forward(self, y, x):
        x_norm = 2 * x / (self._width - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim=2)
        return torch.sin(self.projector(z))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'features': 4, 'height': 4, 'width': 4}]
