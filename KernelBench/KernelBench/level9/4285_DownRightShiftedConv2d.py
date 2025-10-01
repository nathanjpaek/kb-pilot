import torch
import torch.nn as nn


class DownRightShiftedConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_pad = nn.ConstantPad2d((self.kernel_size[1] - 1, 0, self
            .kernel_size[0] - 1, 0), 0.0)

    def forward(self, x):
        x = self.shift_pad(x)
        return super().forward(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
