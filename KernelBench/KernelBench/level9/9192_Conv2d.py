import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.functional
import torch.autograd


def weight_standardization(weight: 'torch.Tensor', eps: 'float'):
    """
    ## Weight Standardization

    $$\\hat{W}_{i,j} = \\frac{W_{i,j} - \\mu_{W_{i,\\cdot}}} {\\sigma_{W_{i,\\cdot}}}$$

    where,

    \\begin{align}
    W &\\in \\mathbb{R}^{O \\times I} \\\\
    \\mu_{W_{i,\\cdot}} &= \\frac{1}{I} \\sum_{j=1}^I W_{i,j} \\\\
    \\sigma_{W_{i,\\cdot}} &= \\sqrt{\\frac{1}{I} \\sum_{j=1}^I W^2_{i,j} - \\mu^2_{W_{i,\\cdot}} + \\epsilon} \\\\
    \\end{align}

    for a 2D-convolution layer $O$ is the number of output channels ($O = C_{out}$)
    and $I$ is the number of input channels times the kernel size ($I = C_{in} \\times k_H \\times k_W$)
    """
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / torch.sqrt(var + eps)
    return weight.view(c_out, c_in, *kernel_shape)


class Conv2d(nn.Conv2d):
    """
    ## 2D Convolution Layer

    This extends the standard 2D Convolution layer and standardize the weights before the convolution step.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups: 'int'=1, bias: 'bool'=True,
        padding_mode: 'str'='zeros', eps: 'float'=1e-05):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            groups, bias=bias, padding_mode=padding_mode)
        self.eps = eps

    def forward(self, x: 'torch.Tensor'):
        return F.conv2d(x, weight_standardization(self.weight, self.eps),
            self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
