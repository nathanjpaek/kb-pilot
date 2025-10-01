import torch
import torch.nn as nn
import torch.utils.cpp_extension


class AdaILN(nn.Module):

    def __init__(self, channels, resl, eps=1e-08):
        super().__init__()
        self.rho = nn.Parameter(torch.Tensor(1, channels, 1, 1))
        self.rho.data.fill_(1.0)
        self.instance_norm = nn.InstanceNorm2d(channels, eps=eps, affine=False)
        self.layer_norm = nn.LayerNorm((channels, resl, resl), eps=eps,
            elementwise_affine=False)

    def forward(self, x, gamma, beta):
        i_norm = self.instance_norm(x)
        l_norm = self.layer_norm(x)
        out = i_norm * self.rho.expand(x.size(0), -1, -1, -1) + l_norm * (1 -
            self.rho.expand(x.size(0), -1, -1, -1))
        out = out * gamma.view(out.size(0), -1, 1, 1) + beta.view(out.size(
            0), -1, 1, 1)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'resl': 4}]
