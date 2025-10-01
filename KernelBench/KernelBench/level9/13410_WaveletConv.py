import torch
import torch.nn as nn


class WaveletConv(nn.Module):

    def __init__(self, in_features, out_features, num_nodes, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = nn.Linear(in_features, out_features, bias=bias)
        self.filter = nn.Parameter(torch.ones(num_nodes, 1))

    def reset_parameters(self):
        self.w.reset_parameters()
        self.filter.data.fill_(1.0)

    def forward(self, x, wavelet, inverse_wavelet):
        h = self.w(x)
        h = inverse_wavelet.mm(h)
        h = self.filter * h
        out = wavelet.mm(h)
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_features}, {self.out_features})'
            )


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4, 'num_nodes': 4}]
