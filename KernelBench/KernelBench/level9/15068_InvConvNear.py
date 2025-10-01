import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data


class InvConvNear(nn.Module):

    def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian
        w_init = torch.qr(torch.FloatTensor(self.n_split, self.n_split).
            normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])
        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c //
            self.n_split, t)
        if reverse:
            if hasattr(self, 'weight_inv'):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float())
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len
        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)
        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def store_inverse(self):
        self.weight_inv = torch.inverse(self.weight.float())


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4}]
