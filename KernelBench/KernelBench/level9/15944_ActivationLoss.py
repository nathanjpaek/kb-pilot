import torch
import torch.utils.data
from torch import nn


class ActivationLoss(nn.Module):

    def __init__(self):
        super(ActivationLoss, self).__init__()

    def forward(self, zero, one, labels):
        loss_act = torch.abs(one - labels.data) + torch.abs(zero - (1.0 -
            labels.data))
        return 1 / labels.shape[0] * loss_act.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
