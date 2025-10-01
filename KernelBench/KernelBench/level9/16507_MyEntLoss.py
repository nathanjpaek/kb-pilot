import torch
import torch.nn as nn


class MyEntLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.nn.Softmax(dim=1)(x)
        p = x / torch.repeat_interleave(x.sum(dim=1).unsqueeze(-1), repeats
            =20, dim=1)
        logp = torch.log2(p)
        ent = -torch.mul(p, logp)
        entloss = torch.sum(ent, dim=1)
        return entloss


def get_inputs():
    return [torch.rand([4, 80, 4, 4])]


def get_init_inputs():
    return [[], {}]
