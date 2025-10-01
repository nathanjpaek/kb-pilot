import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class MyBatchNorm1d(_BatchNorm):
    """Applies Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.
    .. math::
        y = \\frac{x - mean[x]}{ \\sqrt{Var[x] + \\epsilon}} * gamma + beta
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm
    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
        momentum_decay_step=None, momentum_decay=1):
        super(MyBatchNorm1d, self).__init__(num_features, eps, momentum, affine
            )
        self.momentum_decay_step = momentum_decay_step
        self.momentum_decay = momentum_decay
        self.momentum_original = self.momentum

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.
                format(input.dim()))
        super(MyBatchNorm1d, self)._check_input_dim(input)

    def forward(self, input, epoch=None):
        if (epoch is not None and epoch >= 1 and self.momentum_decay_step
             is not None and self.momentum_decay_step > 0):
            self.momentum = self.momentum_original * self.momentum_decay ** (
                epoch // self.momentum_decay_step)
            if self.momentum < 0.01:
                self.momentum = 0.01
        return F.batch_norm(input, self.running_mean, self.running_var,
            self.weight, self.bias, self.training, self.momentum, self.eps)


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return 1.78718727865 * (x * torch.sigmoid(x) - 0.20662096414)


class EquivariantLayer(nn.Module):

    def __init__(self, num_in_channels, num_out_channels, activation='relu',
        normalization=None, momentum=0.1, bn_momentum_decay_step=None,
        bn_momentum_decay=1):
        super(EquivariantLayer, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.activation = activation
        self.normalization = normalization
        self.conv = nn.Conv1d(self.num_in_channels, self.num_out_channels,
            kernel_size=1, stride=1, padding=0)
        if 'batch' == self.normalization:
            self.norm = MyBatchNorm1d(self.num_out_channels, momentum=
                momentum, affine=True, momentum_decay_step=
                bn_momentum_decay_step, momentum_decay=bn_momentum_decay)
        elif 'instance' == self.normalization:
            self.norm = nn.InstanceNorm1d(self.num_out_channels, momentum=
                momentum, affine=True)
        if 'relu' == self.activation:
            self.act = nn.ReLU()
        elif 'elu' == self.activation:
            self.act = nn.ELU(alpha=1.0)
        elif 'swish' == self.activation:
            self.act = Swish()
        elif 'leakyrelu' == self.activation:
            self.act = nn.LeakyReLU(0.1)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, MyBatchNorm1d) or isinstance(m, nn.
                InstanceNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, epoch=None):
        y = self.conv(x)
        if self.normalization == 'batch':
            y = self.norm(y, epoch)
        elif self.normalization is not None:
            y = self.norm(y)
        if self.activation is not None:
            y = self.act(y)
        return y


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_in_channels': 4, 'num_out_channels': 4}]
