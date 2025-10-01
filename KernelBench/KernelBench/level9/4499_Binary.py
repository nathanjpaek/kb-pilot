import torch
import torch.nn as nn


class Binary(nn.Module):

    def __init__(self):
        super().__init__()
        self._criteria = nn.BCELoss()

    def forward(self, output, y):
        y_copy = y.clone()
        y_copy[y > 0] = 0.9
        y_copy[y < 0] = 0
        return self._criteria(output, y_copy)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
