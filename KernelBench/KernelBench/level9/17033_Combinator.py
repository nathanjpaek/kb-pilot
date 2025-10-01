import torch
from torch import nn
import torch.autograd


class Combinator(nn.Module):
    """
    The vanilla combinator function g() that combines vertical and
    lateral connections as explained in Pezeshki et al. (2016).
    The weights are initialized as described in Eq. 17
    and the g() is defined in Eq. 16.
    """

    def __init__(self, n_channels, length, data_type='2d'):
        super(Combinator, self).__init__()
        if data_type == '2d':
            zeros = torch.zeros(n_channels, length, length)
            ones = torch.ones(n_channels, length, length)
        elif data_type == '1d':
            zeros = torch.zeros(n_channels, length)
            ones = torch.ones(n_channels, length)
        else:
            raise ValueError
        self.b0 = nn.Parameter(zeros)
        self.w0z = nn.Parameter(ones)
        self.w0u = nn.Parameter(zeros)
        self.w0zu = nn.Parameter(ones)
        self.b1 = nn.Parameter(zeros)
        self.w1z = nn.Parameter(ones)
        self.w1u = nn.Parameter(zeros)
        self.w1zu = nn.Parameter(zeros)
        self.wsig = nn.Parameter(ones)

    def forward(self, z_tilde, ulplus1):
        assert z_tilde.shape == ulplus1.shape
        out = self.b0 + z_tilde.mul(self.w0z) + ulplus1.mul(self.w0u
            ) + z_tilde.mul(ulplus1.mul(self.w0zu)) + self.wsig.mul(torch.
            sigmoid(self.b1 + z_tilde.mul(self.w1z) + ulplus1.mul(self.w1u) +
            z_tilde.mul(ulplus1.mul(self.w1zu))))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_channels': 4, 'length': 4}]
