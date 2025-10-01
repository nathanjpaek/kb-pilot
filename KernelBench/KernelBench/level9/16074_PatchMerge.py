import torch
from torch import nn


def patchify(input, size):
    batch, height, width, dim = input.shape
    return input.view(batch, height // size, size, width // size, size, dim
        ).permute(0, 1, 3, 2, 4, 5).reshape(batch, height // size, width //
        size, -1)


class PatchMerge(nn.Module):

    def __init__(self, in_dim, out_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(in_dim * window_size * window_size)
        self.linear = nn.Linear(in_dim * window_size * window_size, out_dim,
            bias=False)

    def forward(self, input):
        out = patchify(input, self.window_size)
        out = self.norm(out)
        out = self.linear(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'window_size': 4}]
