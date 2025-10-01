import torch
from torch import nn


class GlobalAvgPool2d(nn.Module):
    """Performs global average pooling over the entire height and width of a batched 2D tensor

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]
            ).view(-1, input.size(1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
