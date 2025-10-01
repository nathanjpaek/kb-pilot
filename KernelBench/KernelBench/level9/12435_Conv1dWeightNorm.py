import torch
import torch.nn as nn


class Conv1dWeightNorm(nn.Module):
    """
    Conv1d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dWeightNorm, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        self.conv = nn.utils.weight_norm(self.conv)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            out = self(x)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-06)
            self.conv.weight_g.mul_(inv_stdv.view(n_channels, 1, 1))
            if self.conv.bias is not None:
                self.conv.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.conv(input)

    def extra_repr(self):
        return self.conv.extra_repr()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
