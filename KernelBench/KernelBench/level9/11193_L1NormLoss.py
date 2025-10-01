import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class L1NormLoss(nn.Module):

    def __init__(self, loss_weight=0.0005, average=True):
        super(L1NormLoss, self).__init__()
        self.loss_weight = loss_weight
        self.average = average

    def forward(self, x1, x2, x3, length):
        loss_norm = (x1 + x2 + x3) / 3
        if self.average:
            loss_norm = loss_norm / length
        return self.loss_weight * loss_norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
