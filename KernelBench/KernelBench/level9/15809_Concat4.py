import torch
import torch.onnx
import torch.nn as nn


class Concat4(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, c0, c1, c2, c3):
        return torch.cat([c0, c1, c2, c3], dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
