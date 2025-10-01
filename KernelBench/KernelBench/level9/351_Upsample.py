import torch
from torch import nn


class Upsample(nn.Module):

    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(input_dim, output_dim,
            kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'kernel': 4, 'stride': 1}]
