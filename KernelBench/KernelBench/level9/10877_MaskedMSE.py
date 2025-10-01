import torch
import torch.nn as nn


class MaskedMSE(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask)
        return self.mse(pred, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
