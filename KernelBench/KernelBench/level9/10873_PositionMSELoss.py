import torch
import torch.nn as nn


class PositionMSELoss(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask):
        pred = torch.mul(pred, mask.unsqueeze(2))
        return self.mse(pred, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
