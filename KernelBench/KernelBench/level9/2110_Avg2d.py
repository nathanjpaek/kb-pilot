import torch
import torch as T


class Avg2d(T.nn.Module):

    def forward(self, x):
        return x.mean((-2, -1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
