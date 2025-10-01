import torch
import torch.nn as nn
import torch.nn.functional as F


class RotatELoss(nn.Module):

    def __init__(self):
        super(RotatELoss, self).__init__()

    def forward(self, p_score, n_score, penalty=None):
        return torch.mean(-F.logsigmoid(p_score) - F.logsigmoid(-n_score))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
