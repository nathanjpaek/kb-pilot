import torch
import torch.nn as nn


class NCCLoss(nn.Module):
    """
    A implementation of the normalized cross correlation (NCC)
    """

    def forward(self, input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        input_minus_mean = input - torch.mean(input, 1).view(input.shape[0], 1)
        target_minus_mean = target - torch.mean(target, 1).view(input.shape
            [0], 1)
        nccSqr = (input_minus_mean * target_minus_mean).mean(1) / torch.sqrt(
            (input_minus_mean ** 2).mean(1) * (target_minus_mean ** 2).mean(1))
        nccSqr = nccSqr.mean()
        return (1 - nccSqr) * input.shape[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
