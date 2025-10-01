import torch
import torch.nn as nn
from collections import OrderedDict


class EncoderBlock(nn.Module):

    def __init__(self, n_in, n_out, n_layers):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hid = self.n_out
        self.n_layers = n_layers
        self.post_gain = 1.0
        self.id_path = nn.Conv2d(self.n_in, self.n_out, 1
            ) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([('conv_1', nn.Conv2d(
            self.n_in, self.n_hid, kernel_size=3, padding=1)), ('relu_1',
            nn.ReLU()), ('conv_2', nn.Conv2d(self.n_hid, self.n_hid,
            kernel_size=3, padding=1)), ('relu_2', nn.ReLU()), ('conv_3',
            nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, padding=1)), (
            'relu_3', nn.ReLU()), ('conv_4', nn.Conv2d(self.n_hid, self.
            n_out, kernel_size=3, padding=1)), ('relu_4', nn.ReLU())]))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4, 'n_layers': 1}]
