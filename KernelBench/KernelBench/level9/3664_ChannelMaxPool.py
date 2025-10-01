import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelMaxPool(nn.MaxPool1d):

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.max_pool1d(input, self.kernel_size, self.stride, self.
            padding, self.dilation, self.ceil_mode, self.return_indices)
        return pooled.permute(0, 2, 1).view(n, 1, w, h)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
