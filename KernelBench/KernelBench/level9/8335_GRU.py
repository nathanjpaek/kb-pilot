import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, outfea):
        super(GRU, self).__init__()
        self.ff = nn.Linear(2 * outfea, 2 * outfea)
        self.zff = nn.Linear(2 * outfea, outfea)
        self.outfea = outfea

    def forward(self, x, xh):
        r, u = torch.split(torch.sigmoid(self.ff(torch.cat([x, xh], -1))),
            self.outfea, -1)
        z = torch.tanh(self.zff(torch.cat([x, r * xh], -1)))
        x = u * z + (1 - u) * xh
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'outfea': 4}]
