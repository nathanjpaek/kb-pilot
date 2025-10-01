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


class Conv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, gain=1,
        use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1
        he_std = gain / (input_channels * output_channels * kernel_size *
            kernel_size) ** 0.5
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_channels,
            input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self
                .b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.
                kernel_size // 2)


class ConvDownsample2d(nn.Module):

    def __init__(self, kernel_size, input_channels, output_channels, k=[1, 
        3, 3, 1], factor=2, gain=1, use_wscale=True, lrmul=1, bias=True):
        """
            ConvDownsample2D method in D_stylegan2.
        :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                  The default is `[1] * factor`, which corresponds to average pooling.
        :param factor: Integer downsampling factor (default: 2).
        :param gain:   Scaling factor for signal magnitude (default: 1.0).

            Returns: Tensor of the shape `[N, C, H // factor, W // factor]`
        """
        super().__init__()
        assert isinstance(factor, int
            ) and factor >= 1, 'factor must be larger than 1! (default: 2)'
        assert kernel_size >= 1 and kernel_size % 2 == 1
        he_std = gain / (input_channels * output_channels * kernel_size *
            kernel_size) ** 0.5
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_channels,
            input_channels, kernel_size, kernel_size) * init_std)
        self.convH, self.convW = self.weight.shape[2:]
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None
        self.gain = gain
        self.factor = factor
        self.k = _setup_kernel(k) * self.gain
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
        self.k = nn.Parameter(self.k, requires_grad=False)
        self.p = self.k.shape[-1] - self.factor + (self.convW - 1)
        self.padx0, self.pady0 = (self.p + 1) // 2, (self.p + 1) // 2
        self.padx1, self.pady1 = self.p // 2, self.p // 2
        self.kernelH, self.kernelW = self.k.shape[2:]

    def forward(self, x):
        y = x.clone()
        y = y.reshape([-1, x.shape[2], x.shape[3], 1])
        inC, inH, inW = x.shape[1:]
        y = torch.reshape(y, (-1, inH, inW, 1))
        y = F.pad(y, (0, 0, max(self.pady0, 0), max(self.pady1, 0), max(
            self.padx0, 0), max(self.padx1, 0), 0, 0))
        y = y[:, max(-self.pady0, 0):y.shape[1] - max(-self.pady1, 0), max(
            -self.padx0, 0):y.shape[2] - max(-self.padx1, 0), :]
        y = y.permute(0, 3, 1, 2)
        y = y.reshape(-1, 1, inH + self.pady0 + self.pady1, inW + self.
            padx0 + self.padx1)
        y = F.conv2d(y, self.k)
        y = y.view(-1, 1, inH + self.pady0 + self.pady1 - self.kernelH + 1,
            inW + self.padx0 + self.padx1 - self.kernelW + 1)
        if inH != y.shape[1]:
            y = F.interpolate(y, size=(inH, inW))
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(-1, inC, inH, inW)
        x1 = F.conv2d(y, self.weight * self.w_lrmul, self.bias * self.
            b_lrmul, stride=self.factor, padding=self.convW // 2)
        out = F.leaky_relu(x1, 0.2, inplace=True)
        out = out * np.sqrt(2)
        return out


class DBlock(nn.Module):
    """
        D_stylegan2 Basic Block.
    """

    def __init__(self, in1, in2, out3, use_wscale=True, lrmul=1,
        resample_kernel=[1, 3, 3, 1], architecture='resnet'):
        super().__init__()
        self.arch = architecture
        self.conv0 = Conv2d(input_channels=in1, output_channels=in2,
            kernel_size=3, use_wscale=use_wscale, lrmul=lrmul, bias=True)
        self.conv1_down = ConvDownsample2d(kernel_size=3, input_channels=
            in2, output_channels=out3, k=resample_kernel)
        self.res_conv2_down = ConvDownsample2d(kernel_size=1,
            input_channels=in1, output_channels=out3, k=resample_kernel)

    def forward(self, x):
        t = x.clone()
        x = self.conv0(x)
        x = self.conv1_down(x)
        if self.arch == 'resnet':
            t = self.res_conv2_down(t)
            x = (x + t) * (1 / np.sqrt(2))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in1': 4, 'in2': 4, 'out3': 4}]
