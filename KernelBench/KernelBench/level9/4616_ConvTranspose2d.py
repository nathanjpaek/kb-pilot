import torch
import torch.nn as nn


class ConvTranspose2d(nn.Module):

    def __init__(self):
        super(ConvTranspose2d, self).__init__()
        self.convtranspose2d = nn.ConvTranspose2d(16, 33, 3, stride=2)

    def forward(self, x):
        x = self.convtranspose2d(x)
        return x


def get_inputs():
    return [torch.rand([4, 16, 4, 4])]


def get_init_inputs():
    return [[], {}]
