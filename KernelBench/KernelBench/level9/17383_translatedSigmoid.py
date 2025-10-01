import torch
from torch import nn


class translatedSigmoid(nn.Module):

    def __init__(self):
        super(translatedSigmoid, self).__init__()
        self.beta = nn.Parameter(torch.tensor([-3.5]))

    def forward(self, x):
        beta = torch.nn.functional.softplus(self.beta)
        alpha = -beta * 6.9077542789816375
        return torch.sigmoid((x + alpha) / beta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
