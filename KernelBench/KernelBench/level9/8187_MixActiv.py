import torch
import torch as th
from torch import nn


def gauss(x, mean=0, std=1):
    return th.exp(-(x - mean) ** 2 / (2 * std ** 2))


class MixActiv(nn.Module):

    def __init__(self):
        super().__init__()
        self.activations = th.sin, th.tanh, gauss, th.relu
        self.n_activs = len(self.activations)

    def forward(self, x):
        n_chan = x.shape[1]
        chans_per_activ = n_chan / self.n_activs
        chan_i = 0
        xs = []
        for i, activ in enumerate(self.activations):
            xs.append(activ(x[:, int(chan_i):int(chan_i + chans_per_activ),
                :, :]))
            chan_i += chans_per_activ
        x = th.cat(xs, axis=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
