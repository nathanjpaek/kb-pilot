import torch
import torch.nn as nn


class Differencial_SMAPE(nn.Module):

    def __init__(self):
        super(Differencial_SMAPE, self).__init__()

    def forward(self, true, predicted):
        epsilon = 0.1
        summ = torch.clamp(torch.abs(true) + torch.abs(predicted) + epsilon,
            min=0.5 + epsilon)
        smape = torch.abs(predicted - true) / summ * 2.0
        return torch.sum(smape)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
