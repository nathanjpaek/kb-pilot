from _paritybench_helpers import _mock_config
import torch
import torch.utils.data
from torch import nn
import torch.nn.parallel
from collections import OrderedDict


class Generator_mnist(nn.Module):

    def __init__(self, opt):
        super(Generator_mnist, self).__init__()
        self.decoder = nn.Sequential(OrderedDict([('deconv1', nn.
            ConvTranspose2d(4, 16, 2, stride=2)), ('relu1', nn.ReLU()), (
            'deconv2', nn.ConvTranspose2d(16, 1, 2, stride=2)), ('sigmoid',
            nn.Sigmoid())]))

    def forward(self, z):
        return self.decoder(z)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'opt': _mock_config()}]
