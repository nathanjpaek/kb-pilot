import torch
from torch import nn


class GlobalMaxPool1d(nn.Module):
    """Performs global max pooling over the entire length of a batched 1D tensor

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]
            ).view(-1, input.size(1))


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
