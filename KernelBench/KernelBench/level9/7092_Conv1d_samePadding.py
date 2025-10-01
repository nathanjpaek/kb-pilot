import torch
from torch import nn
import torch.nn.functional as F


class Conv1d_samePadding(nn.Conv1d):

    def __init__(self, *args, padding: int=0, **kwargs):
        assert padding == 0, "no additional padding on top of 'same' padding"
        kwargs['padding'] = 0
        super().__init__(*args, **kwargs)

    def same_padding_1d(self, input):
        input_duration = input.size(2)
        filter_duration = self.weight.size(2)
        out_duration = (input_duration + self.stride[0] - 1) // self.stride[0]
        padding_duration = max(0, (out_duration - 1) * self.stride[0] + (
            filter_duration - 1) * self.dilation[0] + 1 - input_duration)
        duration_odd = padding_duration % 2
        input = F.pad(input, (padding_duration // 2, padding_duration // 2 +
            int(duration_odd)))
        return input

    def forward(self, input):
        input = self.same_padding_1d(input)
        return super().forward(input)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
