import torch
import torch.nn.parallel
import torch.optim
import torch
import torch.nn as nn


class CONV1d_FusionBlock(nn.Module):

    def __init__(self, in_channels, n_segment, n_div):
        super(CONV1d_FusionBlock, self).__init__()
        self.n_div = n_div
        self.fold = in_channels // n_div
        self.n_segment = n_segment
        self.temporal_conv = nn.Conv3d(in_channels=2 * self.fold,
            out_channels=2 * self.fold, kernel_size=(3, 1, 1), padding=(1, 
            0, 0), stride=1, bias=True)
        nn.init.constant_(self.temporal_conv.weight, 0)
        nn.init.constant_(self.temporal_conv.bias, 0)

    def forward(self, x):
        """
        :param x: (nt, c, h, w)
        :return:(nt, c, h, w)
        """
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)
        out_part = x[:, :2 * self.fold]
        out_part = self.temporal_conv(out_part)
        out = torch.zeros_like(x)
        out[:, :2 * self.fold] = out_part
        out[:, 2 * self.fold:] = x[:, 2 * self.fold:]
        out = out.transpose(1, 2).contiguous().view(nt, c, h, w)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'n_segment': 4, 'n_div': 4}]
