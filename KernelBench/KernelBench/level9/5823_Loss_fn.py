import torch
import torch.nn as nn


class Loss_fn(nn.Module):

    def __init__(self, eps=0.001):
        super().__init__()
        self.eps = eps

    def forward(self, ip, target):
        diff = ip - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
