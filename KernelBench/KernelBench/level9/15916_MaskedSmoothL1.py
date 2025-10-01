import torch
import torch.nn as nn


class MaskedSmoothL1(nn.Module):

    def __init__(self):
        super(MaskedSmoothL1, self).__init__()
        self.criterion = nn.SmoothL1Loss(size_average=True)

    def forward(self, input, target, mask):
        self.loss = self.criterion(input, target * mask)
        return self.loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
