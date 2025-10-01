import torch
import torch.nn as nn
from typing import Counter
from collections import Counter


class NestedNetInnerModule(nn.Module):
    """
    A submodule for the nested net test module below.
    """

    def __init__(self, lin_op: 'str'='addmm') ->None:
        super().__init__()
        conv_input_size = 2, 5
        conv_in = 2
        conv_out = 2
        kernel_size = 1
        padding = 0
        fc_in = 10
        fc_out = 10
        self.conv = nn.Conv1d(in_channels=conv_in, out_channels=conv_out,
            kernel_size=kernel_size, padding=padding)
        self.fc = nn.Linear(in_features=fc_in, out_features=fc_out)
        fc_flops = fc_in * fc_out
        fc_flops = Counter({lin_op: fc_flops})
        spatial_pos = conv_input_size[1] + 2 * padding - 2 * (kernel_size // 2)
        conv_flops = spatial_pos * kernel_size * conv_in * conv_out
        conv_flops = Counter({'conv': conv_flops})
        model_flops = conv_flops + fc_flops
        self.flops = {'': model_flops, 'fc': fc_flops, 'conv': conv_flops}
        self.name_to_module = {'': self, 'fc': self.fc, 'conv': self.conv}

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = x.reshape(-1, 2, 5)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = 3 * self.fc(x) + 1
        return x


def get_inputs():
    return [torch.rand([4, 2, 5])]


def get_init_inputs():
    return [[], {}]
