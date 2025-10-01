import torch
from torch import nn


def _assert_no_grad(tensor):
    assert not tensor.requires_grad


class Corr(nn.Module):

    def __init__(self, eps=1e-12):
        self.eps = eps
        super().__init__()

    def forward(self, output, target):
        _assert_no_grad(target)
        delta_out = output - output.mean(0, keepdim=True).expand_as(output)
        delta_target = target - target.mean(0, keepdim=True).expand_as(target)
        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)
        corrs = (delta_out * delta_target).mean(0, keepdim=True) / ((
            var_out + self.eps) * (var_target + self.eps)).sqrt()
        return corrs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
