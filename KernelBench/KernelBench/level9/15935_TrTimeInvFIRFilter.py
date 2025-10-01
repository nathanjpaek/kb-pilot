import torch
from torch import nn
from torch.nn import functional as F


class TrTimeInvFIRFilter(nn.Conv1d):
    """Trainable Time-invatiant FIR filter implementation

    H(z) = \\sigma_{k=0}^{filt_dim} b_{k}z_{-k}

    Note that b_{0} is fixed to 1 if fixed_0th is True.

    Args:
        channels (int): input channels
        filt_dim (int): FIR filter dimension
        causal (bool): causal
        tanh (bool): apply tanh to filter coef or not.
        fixed_0th (bool): fix the first filt coef to 1 or not.
    """

    def __init__(self, channels, filt_dim, causal=True, tanh=True,
        fixed_0th=True):
        init_filt_coef = torch.randn(filt_dim) * (1 / filt_dim)
        kernel_size = len(init_filt_coef)
        self.causal = causal
        if causal:
            padding = (kernel_size - 1) * 1
        else:
            padding = (kernel_size - 1) // 2 * 1
        super(TrTimeInvFIRFilter, self).__init__(channels, channels,
            kernel_size, padding=padding, groups=channels, bias=None)
        self.weight.data[:, :, :] = init_filt_coef.flip(-1)
        self.weight.requires_grad = True
        self.tanh = tanh
        self.fixed_0th = fixed_0th

    def get_filt_coefs(self):
        b = torch.tanh(self.weight) if self.tanh else self.weight
        b = b.clone()
        if self.fixed_0th:
            b[:, :, -1] = 1
        return b

    def forward(self, x):
        b = self.get_filt_coefs()
        out = F.conv1d(x, b, self.bias, self.stride, self.padding, self.
            dilation, self.groups)
        if self.padding[0] > 0:
            out = out[:, :, :-self.padding[0]] if self.causal else out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'filt_dim': 4}]
