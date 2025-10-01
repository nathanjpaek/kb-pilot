import torch
import torch.utils.data
import torch.nn as nn


def add_layer(seq, ix, n_inputs, n_outputs, nonlin, normalization):
    seq.add_module('L' + str(ix), nn.Linear(n_inputs, n_outputs))
    if ix > 0 and normalization:
        if normalization == 'LN':
            seq.main.add_module('A' + str(ix), nn.LayerNorm(n_outputs))
        else:
            raise ValueError('Unknown normalization: {}'.format(normalization))
    if nonlin == 'LeakyReLU':
        seq.add_module('N' + str(ix), nn.LeakyReLU(0.2, inplace=True))
    elif nonlin == 'ReLU':
        seq.add_module('N' + str(ix), nn.ReLU(inplace=True))
    elif nonlin == 'Sigmoid':
        seq.add_module('N' + str(ix), nn.Sigmoid())


class D_phiVpsi(nn.Module):

    def __init__(self, insizes=[1, 1], layerSizes=[[32, 32, 16]] * 2,
        nonlin='LeakyReLU', normalization=None):
        super(D_phiVpsi, self).__init__()
        self.phi_x, self.psi_y = nn.Sequential(), nn.Sequential()
        for seq, insize, layerSize in [(self.phi_x, insizes[0], layerSizes[
            0]), (self.psi_y, insizes[1], layerSizes[1])]:
            for ix, n_inputs, n_outputs in zip(range(len(layerSize)), [
                insize] + layerSize[:-1], layerSize):
                add_layer(seq, ix, n_inputs, n_outputs, nonlin, normalization)
        self.phiD, self.psiD = layerSizes[0][-1], layerSizes[1][-1]
        self.W = nn.Parameter(torch.randn(self.phiD, self.psiD))

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y = y.view(x.size(0), 1)
        phi_x = self.phi_x(x)
        psi_y = self.psi_y(y)
        out = (torch.mm(phi_x, self.W) * psi_y).sum(1, keepdim=True)
        return out


def get_inputs():
    return [torch.rand([4, 1]), torch.rand([4, 1])]


def get_init_inputs():
    return [[], {}]
