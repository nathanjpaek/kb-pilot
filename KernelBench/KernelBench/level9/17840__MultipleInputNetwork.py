import torch
import torch.nn as _nn


class _MultipleInputNetwork(_nn.Module):

    def __init__(self):
        super(_MultipleInputNetwork, self).__init__()
        self.conv = _nn.Conv2d(3, 16, 3)

    def forward(self, inp1, inp2):
        inp = inp1 * inp2
        out = self.conv(inp)
        return out


def get_inputs():
    return [torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
