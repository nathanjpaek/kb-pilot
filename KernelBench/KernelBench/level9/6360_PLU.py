import torch
import torch.nn as nn


class PLU(nn.Module):

    def __init__(self):
        super(PLU, self).__init__()
        self.w1 = torch.nn.Parameter(torch.ones(1))
        self.w2 = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.w1 * torch.max(x, torch.zeros_like(x)
            ) + self.w2 * torch.min(x, torch.zeros_like(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
