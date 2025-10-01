import torch
import torch.nn as nn


class CrossEntropyWithLogSoftmax(nn.Module):

    def forward(self, y_hat, y):
        return -(y_hat * y).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
