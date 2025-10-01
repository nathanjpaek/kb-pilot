import torch
import torch.nn as nn


class ReSentenceMatrixLayer(nn.Module):

    def __init__(self, in_size, out_size=1):
        super(ReSentenceMatrixLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.a_Asem = nn.Parameter(torch.tensor(0.0))
        self.linear = nn.Linear(in_size * 2, out_size)

    def forward(self, x, adj):
        xi = x.unsqueeze(-2)
        xi = xi.expand(xi.shape[0], xi.shape[1], xi.shape[1], xi.shape[-1])
        xj = x.unsqueeze(1)
        xj = xj.expand(xj.shape[0], xj.shape[2], xj.shape[2], xj.shape[-1])
        xij = torch.cat((xi, xj), -1)
        A_esm = torch.sigmoid(self.linear(xij).squeeze()
            ) + self.a_Asem * adj.to_dense()
        return A_esm


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
