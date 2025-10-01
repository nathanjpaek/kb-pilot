import torch
import torch.nn.parallel
import torch.optim
import torch
import torch.nn as nn


class MOTION_ReplaceBlock_B(nn.Module):
    """
    using diff
    """

    def __init__(self, in_channels, n_segment, n_div):
        super(MOTION_ReplaceBlock_B, self).__init__()
        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment

    def forward(self, x):
        """
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        """
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        out = torch.zeros_like(x)
        out[:, :-1, :self.fold] = x[:, 1:, :self.fold] - x[:, :-1, :self.fold]
        out[:, 1:, self.fold:2 * self.fold] = x[:, 1:, self.fold:2 * self.fold
            ] - x[:, :-1, self.fold:2 * self.fold]
        out[:, :, 2 * self.fold:] = x[:, :, 2 * self.fold:]
        out = out.view(nt, c, h, w)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'n_segment': 4, 'n_div': 4}]
