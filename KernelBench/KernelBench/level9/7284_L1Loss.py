import torch
from torch import nn
from torch import torch


class L1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Yp, Yt):
        num = Yt.size(0)
        Yp = Yp.view(num, -1)
        Yt = Yt.view(num, -1)
        loss = nn.functional.l1_loss(Yp, Yt)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
