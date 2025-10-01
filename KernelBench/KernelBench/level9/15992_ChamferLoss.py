import torch
import torch.nn as nn


class ChamferLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        :param x: (bs, np, 3)
        :param y: (bs, np, 3)
        :return: loss
        """
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        dist = torch.sqrt(1e-06 + torch.sum(torch.pow(x - y, 2), 3))
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)
        return min1.mean() + min2.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
