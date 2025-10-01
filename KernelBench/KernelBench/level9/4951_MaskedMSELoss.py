import torch
import torch.utils.data
from torch import nn


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, output_lengths):
        squared_error = (target - pred) ** 2
        loss = (squared_error.mean(1).sum(1) / output_lengths).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
