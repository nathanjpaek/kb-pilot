import torch
import torch.nn as nn
import torch.utils.data


class TorchGloVeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.reduction = 'sum'

    def forward(self, diffs, weights):
        return torch.sum(0.5 * torch.mul(weights, diffs ** 2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
