import torch
from itertools import product as product
import torch.nn.functional as F
from torch import nn
import torch.optim
import torch.utils.data


class BinQuant(torch.autograd.Function):
    """BinaryConnect quantization.
    Refer:
        https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845/3
    """

    @staticmethod
    def forward(ctx, w):
        """Require w be in range of [0, 1].
        Otherwise, it is not in activate range.
        """
        return w.sign()

    @staticmethod
    def backward(ctx, grad_o):
        grad_i = grad_o.clone()
        return grad_i


class BinConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args):
        self.weight_q = BinQuant.apply(self.weight)
        y = F.conv2d(x, self.weight_q, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
