import torch
import torch.nn as nn


class MaskedSoftL1(nn.Module):
    loss = nn.SmoothL1Loss()

    def __init__(self, factor=5):
        super().__init__()
        self.factor = factor

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask)
        return self.loss(pred / self.factor, target / self.factor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
