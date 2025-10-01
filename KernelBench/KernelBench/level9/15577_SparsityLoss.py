import torch
import torch.nn as nn
import torch.utils.data


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """

    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(
            kernel))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
