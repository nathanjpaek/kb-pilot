import torch
import torch.nn as nn


def squash(x, dim=2):
    v_length_sq = x.pow(2).sum(dim=dim, keepdim=True)
    v_length = torch.sqrt(v_length_sq)
    scaling_factor = v_length_sq / (1 + v_length_sq) / v_length
    return x * scaling_factor


class PrimaryCaps(nn.Module):
    """
    PrimaryCaps layers.
    """

    def __init__(self, in_channels, out_capsules, out_capsule_dim,
        kernel_size=9, stride=2):
        super(PrimaryCaps, self).__init__()
        self.in_channels = in_channels
        self.out_capsules = out_capsules
        self.out_capsule_dim = out_capsule_dim
        self.capsules = nn.Conv2d(in_channels=in_channels, out_channels=
            out_capsules * out_capsule_dim, kernel_size=kernel_size, stride
            =stride, bias=True)

    def forward(self, x):
        """
        Revise based on adambielski's implementation.
        ref: https://github.com/adambielski/CapsNet-pytorch/blob/master/net.py
        """
        batch_size = x.size(0)
        out = self.capsules(x)
        _, _C, H, W = out.size()
        out = out.view(batch_size, self.out_capsules, self.out_capsule_dim,
            H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out, dim=2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_capsules': 4, 'out_capsule_dim': 4}]
