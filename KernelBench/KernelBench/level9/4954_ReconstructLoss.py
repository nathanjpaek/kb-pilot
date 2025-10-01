import torch
import torch.nn as nn


class ReconstructLoss(nn.Module):

    def __init__(self):
        super(ReconstructLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        loss = self.criterion(x, y)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
