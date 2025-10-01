import torch
import torch.nn as nn


class SentenceMatrixLayer(nn.Module):

    def __init__(self, in_size, out_size=1, p_Asem=0.8):
        super(SentenceMatrixLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.p_Asem = p_Asem
        self.linear = nn.Linear(in_size * 2, out_size)

    def forward(self, x, adj):
        xi = x.unsqueeze(-2)
        xi = xi.expand(xi.shape[0], xi.shape[1], xi.shape[1], xi.shape[-1])
        xj = x.unsqueeze(1)
        xj = xj.expand(xj.shape[0], xj.shape[2], xj.shape[2], xj.shape[-1])
        xij = torch.cat((xi, xj), -1)
        A_esm = self.p_Asem * torch.sigmoid(self.linear(xij).squeeze()) + (
            1 - self.p_Asem) * adj
        return A_esm


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4}]
