import torch
from torch import nn


class co_peak_loss(nn.Module):

    def __init__(self):
        super(co_peak_loss, self).__init__()

    def forward(self, co_peak_value):
        a = -1 * co_peak_value
        b = torch.max(torch.zeros_like(co_peak_value), a)
        t = b + torch.log(torch.exp(-b) + torch.exp(a - b))
        loss = torch.mean(t)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
