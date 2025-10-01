import torch
import torch.nn as nn
import torch.distributions
import torch.utils.data


class BinaryLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output):
        return torch.logaddexp(torch.tensor([1.0], device=output.device), -
            output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
