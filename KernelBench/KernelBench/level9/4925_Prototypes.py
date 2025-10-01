import torch
import torch.nn as nn
from torch.nn import functional as F


class Prototypes(nn.Module):

    def __init__(self, fdim, num_classes, temp=0.05):
        super().__init__()
        self.prototypes = nn.Linear(fdim, num_classes, bias=False)
        self.temp = temp

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = self.prototypes(x)
        out = out / self.temp
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'fdim': 4, 'num_classes': 4}]
