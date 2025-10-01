import torch
import torch.nn.parallel
import torch.optim
import torch
import torch.nn as nn


class SELECT_fusion_block(nn.Module):

    def __init__(self, in_channels, n_segment, n_div):
        super(SELECT_fusion_block, self).__init__()
        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment
        self.select_op = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.fusion_conv = nn.Conv2d(in_channels=3 * self.fold,
            out_channels=self.fold, kernel_size=1, padding=0, stride=1,
            bias=True)
        nn.init.constant_(self.fusion_conv.weight, 0)
        nn.init.constant_(self.fusion_conv.bias, 0)

    def forward(self, x):
        """
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        """
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        out = torch.zeros_like(x)
        out_part = x.view(nt, c, h, w)[:, :self.fold]
        out_part_select = self.select_op(out_part)
        out_part_select = out_part_select.view(n_batch, self.n_segment,
            self.fold, h, w)
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        select_left = torch.zeros_like(out_part_select)
        select_right = torch.zeros_like(out_part_select)
        select_left[:, 1:] = out_part_select[:, :-1]
        select_right[:, :-1] = out_part_select[:, 1:]
        out_part = torch.cat([select_left, out_part, select_right], dim=2)
        out_part = out_part.view(nt, -1, h, w)
        out_part = self.fusion_conv(out_part)
        out_part = out_part.view(n_batch, self.n_segment, self.fold, h, w)
        out[:, :, :self.fold] = out_part[:, :, :self.fold]
        out[:, :, self.fold:] = x[:, :, self.fold:]
        out = out.view(nt, c, h, w)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'n_segment': 4, 'n_div': 4}]
