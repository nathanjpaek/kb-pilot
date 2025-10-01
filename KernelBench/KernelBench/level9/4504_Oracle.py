import torch
import torch.nn as nn


class Oracle(nn.Module):

    def __init__(self):
        super().__init__()
        self._criteria = nn.CrossEntropyLoss()

    def forward(self, output, y):
        y_copy = y.clone()
        y_copy[:, 0] += 0.005
        return self._criteria(output, y_copy.argmax(dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
