import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedThing(nn.Module):
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    def forward(self, pred, target, mask):
        pred = torch.log1p(F.relu(pred))
        target = torch.log1p(F.relu(target))
        pred = torch.mul(pred, mask)
        target = torch.mul(target, mask)
        return self.mse(pred, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
