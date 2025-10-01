import torch
import torch.nn as nn


class GramMatrix(nn.Module):

    def forward(self, input):
        n_batches, n_channels, height, width = input.size()
        flattened = input.view(n_batches, n_channels, height * width)
        return torch.bmm(flattened, flattened.transpose(1, 2)).div_(height *
            width)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
