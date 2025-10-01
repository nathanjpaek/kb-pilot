import torch
import torch.nn as nn


class GramMatrix(nn.Module):

    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h * w)
        G = torch.bmm(f, f.transpose(1, 2))
        return G.div_(c * h * w)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
