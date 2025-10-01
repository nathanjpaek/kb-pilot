from _paritybench_helpers import _mock_config
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


class Upsample2d(nn.Module):

    def __init__(self, opts, k=[1, 3, 3, 1], factor=2, down=1, gain=1):
        """
            Upsample2d method in G_synthesis_stylegan2.
        :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                  The default is `[1] * factor`, which corresponds to average pooling.
        :param factor: Integer downsampling factor (default: 2).
        :param gain:   Scaling factor for signal magnitude (default: 1.0).

            Returns: Tensor of the shape `[N, C, H // factor, W // factor]`
        """
        super().__init__()
        assert isinstance(factor, int
            ) and factor >= 1, 'factor must be larger than 1! (default: 2)'
        self.gain = gain
        self.factor = factor
        self.opts = opts
        self.k = _setup_kernel(k) * (self.gain * factor ** 2)
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
        self.k = nn.Parameter(self.k, requires_grad=False)
        self.p = self.k.shape[0] - self.factor
        self.padx0, self.pady0 = (self.p + 1) // 2 + factor - 1, (self.p + 1
            ) // 2 + factor - 1
        self.padx1, self.pady1 = self.p // 2, self.p // 2
        self.kernelH, self.kernelW = self.k.shape[2:]
        self.down = down

    def forward(self, x):
        y = x.clone()
        y = y.reshape([-1, x.shape[2], x.shape[3], 1])
        inC, inH, inW = x.shape[1:]
        y = torch.reshape(y, (-1, inH, 1, inW, 1, 1))
        y = F.pad(y, (0, 0, self.factor - 1, 0, 0, 0, self.factor - 1, 0, 0,
            0, 0, 0))
        y = torch.reshape(y, (-1, 1, inH * self.factor, inW * self.factor))
        y = F.pad(y, (0, 0, max(self.pady0, 0), max(self.pady1, 0), max(
            self.padx0, 0), max(self.padx1, 0), 0, 0))
        y = y[:, max(-self.pady0, 0):y.shape[1] - max(-self.pady1, 0), max(
            -self.padx0, 0):y.shape[2] - max(-self.padx1, 0), :]
        y = y.permute(0, 3, 1, 2)
        y = y.reshape(-1, 1, inH * self.factor + self.pady0 + self.pady1, 
            inW * self.factor + self.padx0 + self.padx1)
        y = F.conv2d(y, self.k)
        y = y.view(-1, 1, inH * self.factor + self.pady0 + self.pady1 -
            self.kernelH + 1, inW * self.factor + self.padx0 + self.padx1 -
            self.kernelW + 1)
        if inH * self.factor != y.shape[1]:
            y = F.interpolate(y, size=(inH * self.factor, inW * self.factor))
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(-1, inC, inH * self.factor, inW * self.factor)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'opts': _mock_config()}]
