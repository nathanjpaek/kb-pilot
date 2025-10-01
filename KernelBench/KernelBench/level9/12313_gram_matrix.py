import torch
import torch.nn as nn


class gram_matrix(nn.Module):

    def forward(self, input):
        b, c, w, h = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
