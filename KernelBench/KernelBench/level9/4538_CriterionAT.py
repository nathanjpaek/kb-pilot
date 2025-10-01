import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._utils
import torch.optim


def at(x):
    return F.normalize(x.pow(2).mean(0).reshape(1, -1), dim=1)


class CriterionAT(nn.Module):

    def __init__(self):
        super(CriterionAT, self).__init__()
        self.at = at

    def forward(self, fs, ft):
        n = ft.size(0)
        loss = sum([(self.at(x) - self.at(y)).pow(2).sum() for x, y in zip(
            fs, ft)]) / n
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
