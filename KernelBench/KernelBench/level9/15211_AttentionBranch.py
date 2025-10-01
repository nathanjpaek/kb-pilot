import torch
import torch.nn as nn


class AttentionBranch(nn.Module):
    """Attention Branch."""

    def __init__(self, nf, k_size=3):
        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1
            ) // 2, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1
            ) // 2, bias=False)
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1
            ) // 2, bias=False)

    def forward(self, x):
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nf': 4}]
