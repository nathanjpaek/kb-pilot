import torch
import torch.nn as nn


class SharpenSoftmax(nn.Module):

    def __init__(self, tau, dim=0):
        super().__init__()
        self.tau = tau
        self.dim = dim

    def forward(self, pred):
        pred = pred / self.tau
        return pred.log_softmax(self.dim)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'tau': 4}]
