import torch
import torch.nn as nn
import torch.distributions
import torch.utils.data


class SavageLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output):
        return 1 / (1 + output.exp()) ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
