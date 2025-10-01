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


class D_concat(nn.Module):

    def __init__(self, insizes=[1, 1], layerSizes=[32, 32, 16], nonlin=
        'LeakyReLU', normalization=None):
        super(D_concat, self).__init__()
        insize = sum(insizes)
        self.main = nn.Sequential()
        for ix, n_inputs, n_outputs in zip(range(len(layerSizes)), [insize] +
            layerSizes[:-1], layerSizes):
            add_layer(self.main, ix, n_inputs, n_outputs, nonlin, normalization
                )
            self.PhiD = n_outputs
        self.V = nn.Linear(self.PhiD, 1, bias=False)
        self.V.weight.data *= 100

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y = y.view(x.size(0), 1)
        inp = torch.cat([x, y], dim=1)
        phi = self.main(inp)
        return self.V(phi)


def get_inputs():
    return [torch.rand([4, 1]), torch.rand([4, 1])]


def get_init_inputs():
    return [[], {}]
