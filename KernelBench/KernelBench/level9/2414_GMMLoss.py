import torch
import numpy as np
import torch.nn as nn


class GMMLoss(nn.Module):

    def __init__(self):
        super(GMMLoss, self).__init__()

    def forward(self, x, mu, std, pi):
        x = x.unsqueeze(-1)
        distrib = torch.exp(-((x - mu) / std) ** 2 / 2) / (std * np.sqrt(2 *
            np.pi))
        distrib = torch.sum(pi * distrib, dim=3)
        loss = -torch.log(distrib).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
