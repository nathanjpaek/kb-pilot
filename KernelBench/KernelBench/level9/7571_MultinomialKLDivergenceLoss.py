import torch
from torch import nn


class MultinomialKLDivergenceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, p_proba, q_proba):
        loss = q_proba * (torch.log(q_proba) - torch.log(p_proba))
        loss = torch.sum(loss)
        return loss / (p_proba.size(1) * p_proba.size(0))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
