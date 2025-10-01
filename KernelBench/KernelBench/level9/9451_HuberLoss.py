import torch
import torch.nn as nn


class HuberLoss(nn.Module):

    def __init__(self, delta=1):
        super().__init__()
        self.delta = delta

    def forward(self, sr, hr):
        l1 = torch.abs(sr - hr)
        mask = l1 < self.delta
        sq_loss = 0.5 * l1 ** 2
        abs_loss = self.delta * (l1 - 0.5 * self.delta)
        return torch.mean(mask * sq_loss + ~mask * abs_loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
