import torch
from torch.nn.modules.loss import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *


class ConvRelu(nn.Module):
    """3x3 convolution followed by ReLU activation building block.
    """

    def __init__(self, num_in, num_out):
        """Creates a `ConvReLU` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
            bias=False)

    def forward(self, x):
        """
        The networks forward pass for which
            autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """
        return F.relu(self.block(x), inplace=True)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in': 4, 'num_out': 4}]
