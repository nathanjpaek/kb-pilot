import torch
from torch import nn


class PositiveLinear(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int') ->None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.softplus = nn.Softplus()

    def forward(self, input: 'torch.Tensor'):
        return input @ self.softplus(self.weight) + self.softplus(self.bias)


class SNRNetwork(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)
        self.gamma_min = nn.Parameter(torch.tensor(-10.0))
        self.gamma_max = nn.Parameter(torch.tensor(20.0))
        self.softplus = nn.Softplus()

    def forward(self, t: 'torch.Tensor'):
        t = torch.cat([torch.tensor([0.0, 1.0], device=t.device), t])
        l1 = self.l1(t[:, None])
        l2 = torch.sigmoid(self.l2(l1))
        l3 = torch.squeeze(l1 + self.l3(l2), dim=-1)
        s0, s1, sched = l3[0], l3[1], l3[2:]
        norm_nlogsnr = (sched - s0) / (s1 - s0)
        nlogsnr = self.gamma_min + self.softplus(self.gamma_max) * norm_nlogsnr
        return -nlogsnr, norm_nlogsnr


def get_inputs():
    return [torch.rand([4])]


def get_init_inputs():
    return [[], {}]
