import torch
import torch.nn as nn


class SimSiamLoss(nn.Module):
    """
    Loss function defined in https://arxiv.org/abs/2011.10566
    """

    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, zx, zy, px, py):
        loss = -(zx.detach() * py).sum(dim=1).mean()
        loss += -(zy.detach() * px).sum(dim=1).mean()
        return loss / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
