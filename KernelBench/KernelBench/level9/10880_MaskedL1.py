import torch
import torch.nn as nn


class MaskedL1(nn.Module):
    l1 = nn.L1Loss()

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.l1(pred, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
