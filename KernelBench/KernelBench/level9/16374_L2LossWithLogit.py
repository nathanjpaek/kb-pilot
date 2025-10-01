import torch
import torch.utils.data
import torch
from torch import nn


class L2LossWithLogit(nn.Module):

    def __init__(self):
        super(L2LossWithLogit, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        return self.mse(p, targets)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
