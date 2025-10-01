import torch
from torch import nn
import torch.optim


class LSGanLoss(nn.Module):

    def __init__(self, layer=3):
        super(LSGanLoss, self).__init__()
        self.layer = layer

    def forward(self, real, fake):
        loss_G = 0
        loss_D = 0
        for i in range(self.layer):
            loss_G = loss_G + torch.mean((fake[i] - torch.ones_like(fake[i]
                )) ** 2)
            loss_D = loss_D + 0.5 * (torch.mean((fake[i] - torch.zeros_like
                (fake[i])) ** 2) + torch.mean((real[i] - torch.ones_like(
                real[i])) ** 2))
        return loss_G, loss_D


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
