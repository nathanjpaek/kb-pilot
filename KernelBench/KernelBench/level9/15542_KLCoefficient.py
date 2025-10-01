import torch
import torch.nn as nn
import torch.nn.functional as F


class KLCoefficient(nn.Module):

    def __init__(self):
        super(KLCoefficient, self).__init__()

    def forward(self, hist1, hist2):
        kl = F.kl_div(hist1, hist2)
        dist = 1.0 / 1 + kl
        return dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
