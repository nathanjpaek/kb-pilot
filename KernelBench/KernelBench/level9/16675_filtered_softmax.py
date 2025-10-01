import torch
import torch.nn as nn


class filtered_softmax(nn.Module):

    def __init__(self):
        super(filtered_softmax, self).__init__()

    def forward(self, x, label):
        x = torch.softmax(x, dim=1)
        x = x * label
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
