import torch
import torch.nn as nn
from torch.nn import functional as F


class PANNsLoss(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def stabilize_input(cwo):
        cwo = torch.where(torch.isnan(cwo), torch.zeros_like(cwo), cwo)
        cwo = torch.where(torch.isinf(cwo), torch.zeros_like(cwo), cwo)
        cwo = cwo.clamp(0, 1)
        return cwo

    def forward(self, input, target):
        cwo = self.stabilize_input(input)
        return F.binary_cross_entropy(cwo, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
