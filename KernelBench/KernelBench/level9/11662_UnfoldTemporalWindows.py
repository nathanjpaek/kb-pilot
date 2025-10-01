import torch
import torch.nn as nn


class UnfoldTemporalWindows(nn.Module):

    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation
        self.padding = (window_size + (window_size - 1) * (window_dilation -
            1) - 1) // 2
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1), dilation
            =(self.window_dilation, 1), stride=(self.window_stride, 1),
            padding=(self.padding, 0))

    def forward(self, x):
        N, C, _T, V = x.shape
        x = self.unfold(x)
        x = x.view(N, C, self.window_size, -1, V).permute(0, 1, 3, 2, 4
            ).contiguous()
        x = x.view(N, C, -1, self.window_size * V)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'window_size': 4, 'window_stride': 1}]
