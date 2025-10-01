import torch
import torch.nn as nn


class RMSPE(nn.Module):

    def __init__(self, eps: 'float'=1e-08):
        super().__init__()
        self.eps = eps

    def forward(self, pred: 'torch.Tensor', target: 'torch.Tensor'):
        return torch.sqrt(torch.mean(torch.square((pred - target).abs() / (
            target.abs() + self.eps))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
