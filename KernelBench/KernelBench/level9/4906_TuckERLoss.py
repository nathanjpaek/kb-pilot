import torch
import torch.nn as nn


class TuckERLoss(nn.Module):

    def __init__(self, margin):
        super(TuckERLoss, self).__init__()
        pass

    def forward(self, p_score, n_score, penalty=None):
        p_score = -torch.mean(torch.log(p_score))
        n_score = -torch.mean(torch.log(1 - n_score))
        return (p_score + n_score) / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
