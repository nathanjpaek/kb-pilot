import torch
import torch.nn as nn


class dense_warp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, h1, cost):
        g2 = torch.zeros_like(h1)
        clone_h1 = h1.detach()
        if h1.device.type == 'cuda':
            g2 = g2
            clone_h1 = clone_h1
        for d in range(cost.size()[-3]):
            g2[:, :, :, 0:cost.size()[-1] - d] += cost[:, d:d + 1, :, 0:
                cost.size()[-1] - d].mul(clone_h1[:, :, :, d:cost.size()[-1]])
        return g2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
