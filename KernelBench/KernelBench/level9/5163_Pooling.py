import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class ReLUConvBN(nn.Module):
    """
    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    padding: int
        zero-padding added to both sides of the input
    dilation: int
        spacing between kernel elements
    bn_affine: bool
        If set to ``True``, ``torch.nn.BatchNorm2d`` will have learnable affine parameters. Default: True
    bn_momentun: float
        the value used for the running_mean and running_var computation. Default: 0.1
    bn_track_running_stats: bool
        When set to ``True``, ``torch.nn.BatchNorm2d`` tracks the running mean and variance. Default: True
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation,
        bn_affine=True, bn_momentum=0.1, bn_track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in,
            C_out, kernel_size, stride=stride, padding=padding, dilation=
            dilation, bias=False), nn.BatchNorm2d(C_out, affine=bn_affine,
            momentum=bn_momentum, track_running_stats=bn_track_running_stats))

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        return self.op(x)


class Pooling(nn.Module):
    """
    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    bn_affine: bool
        If set to ``True``, ``torch.nn.BatchNorm2d`` will have learnable affine parameters. Default: True
    bn_momentun: float
        the value used for the running_mean and running_var computation. Default: 0.1
    bn_track_running_stats: bool
        When set to ``True``, ``torch.nn.BatchNorm2d`` tracks the running mean and variance. Default: True
    """

    def __init__(self, C_in, C_out, stride, bn_affine=True, bn_momentum=0.1,
        bn_track_running_stats=True):
        super(Pooling, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 0, bn_affine,
                bn_momentum, bn_track_running_stats)
        self.op = nn.AvgPool2d(3, stride=stride, padding=1,
            count_include_pad=False)

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        if self.preprocess:
            x = self.preprocess(x)
        return self.op(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'C_in': 4, 'C_out': 4, 'stride': 1}]
