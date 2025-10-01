import torch
from torch import nn


def _assert_no_grad(tensor):
    assert not tensor.requires_grad


class ExpMSE(nn.Module):

    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward(self, output, target):
        _assert_no_grad(target)
        loss = (output - target).pow(2)
        exp_loss = loss * torch.exp(self.lam * loss)
        return exp_loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lam': 4}]
