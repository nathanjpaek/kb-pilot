import torch
import torch.nn as nn


class MaskedLoss(nn.Module):
    mse = nn.MSELoss()

    def forward(self, pred, target, mask):
        pred = torch.log1p(pred).contiguous().view(-1)
        target = torch.log1p(target).contiguous().view(-1)
        mask = mask.view(-1)
        pred = (mask * pred.T).T
        return self.mse(pred, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
