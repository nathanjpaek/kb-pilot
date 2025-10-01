import torch
import torch.nn as nn


class NormalizeImages(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-07
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
            expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1
            ).expand_as(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
