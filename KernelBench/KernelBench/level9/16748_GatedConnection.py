import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class GatedConnection(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Linear(d_model * 2, d_model, True)

    def forward(self, t1, t2):
        g = F.sigmoid(self.w(torch.cat([t1, t2], -1)))
        return g * t1 + (1 - g) * t2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
