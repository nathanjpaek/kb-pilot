import torch
import torch.nn as nn


class ShortcutLayer(nn.Module):

    def __init__(self, idx):
        super(ShortcutLayer, self).__init__()
        self.idx = idx

    def forward(self, x, outputs):
        return x + outputs[self.idx]


def get_inputs():
    return [torch.rand([5, 4, 4, 4]), torch.rand([5, 4, 4, 4])]


def get_init_inputs():
    return [[], {'idx': 4}]
