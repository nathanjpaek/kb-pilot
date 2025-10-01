import torch
import torch as T


class Max2d(T.nn.Module):

    def forward(self, x):
        return x.view(*x.shape[:-2], -1).max(-1)[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
