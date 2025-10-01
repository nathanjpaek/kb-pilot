import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, h_dim, n_chan, out_dim):
        super(Decoder, self).__init__()
        self.h_dim = h_dim
        self.n_chan = n_chan
        self.decoding = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = torch.stack([self.decoding(x[i]) for i in range(self.n_chan)], 0)
        x = x.transpose(0, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'h_dim': 4, 'n_chan': 4, 'out_dim': 4}]
