import torch
import torch.distributed
import torch.nn as nn


class MVloss(nn.Module):

    def __init__(self):
        super(MVloss, self).__init__()

    def forward(self, xRA0, xRA20, xRA_20, target, wRA0, wRA20, wRA_20):
        criterion_MV = torch.nn.CrossEntropyLoss()
        loss_multiview = criterion_MV(wRA0 * xRA0 + wRA20 * xRA20 + wRA_20 *
            xRA_20, target)
        return loss_multiview


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]),
        torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
