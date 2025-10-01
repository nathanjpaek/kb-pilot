import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAvgPool(nn.AvgPool1d):

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.avg_pool1d(input, self.kernel_size, self.stride, self.
            padding, self.ceil_mode, self.count_include_pad)
        return pooled.permute(0, 2, 1).view(n, 1, w, h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
