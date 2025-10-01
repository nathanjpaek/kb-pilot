import torch
from torch import nn as nn
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable as Variable
import torch.utils.data
import torch._utils


class My_loss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) *
            torch.sqrt(torch.sum(torch.pow(vy, 2))))
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) +
            torch.pow(x_m - y_m, 2))
        return -ccc


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
