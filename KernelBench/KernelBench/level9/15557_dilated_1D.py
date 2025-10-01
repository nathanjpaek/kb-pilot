import torch
import torch.utils.data
import torch.nn as nn


class dilated_1D(nn.Module):

    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor)
            )

    def forward(self, input):
        x = self.tconv(input)
        return x


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'cin': 4, 'cout': 4}]
