import torch
import torch.nn as nn
import torch.optim
import torch.cuda


class cha_loss(nn.Module):

    def __init__(self, eps=0.001):
        super(cha_loss, self).__init__()
        self.eps = eps
        return

    def forward(self, inp, target):
        diff = torch.abs(inp - target) ** 2 + self.eps ** 2
        out = torch.sqrt(diff)
        loss = torch.mean(out)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
