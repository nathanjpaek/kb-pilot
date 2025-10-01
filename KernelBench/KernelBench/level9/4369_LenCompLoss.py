import torch
import torch.utils.data
import torch
import torch.nn as nn


class LenCompLoss(nn.Module):

    def __init__(self):
        super(LenCompLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        loss = self.loss(torch.sum(x), torch.sum(y))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
