import torch
import torch.nn as nn
import torch.utils.data
from abc import abstractmethod
from typing import Tuple
import torch.nn


class EfficientBlockBase(nn.Module):
    """
    PyTorchVideo/accelerator provides a set of efficient blocks
    that have optimal efficiency for each target hardware device.

    Each efficient block has two forms:
    - original form: this form is for training. When efficient block is instantiated,
        it is in this original form.
    - deployable form: this form is for deployment. Once the network is ready for
        deploy, it can be converted into deployable form for efficient execution
        on target hardware. One block is transformed into deployable form by calling
        convert() method. By conversion to deployable form,
        various optimization (operator fuse, kernel optimization, etc.) are applied.

    EfficientBlockBase is the base class for efficient blocks.
    All efficient blocks should inherit this base class
    and implement following methods:
    - forward(): same as required by nn.Module
    - convert(): called to convert block into deployable form
    """

    @abstractmethod
    def convert(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class AdaptiveAvgPool3dOutSize1(EfficientBlockBase):
    """
    Implements AdaptiveAvgPool3d with output (T, H, W) = (1, 1, 1). This operator has
    better efficiency than AdaptiveAvgPool for mobile CPU.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.convert_flag = False

    def convert(self, input_blob_size: 'Tuple', **kwargs):
        """
        Converts AdaptiveAvgPool into AvgPool with constant kernel size for better
        efficiency.
        Args:
            input_blob_size (tuple): blob size at the input of
                AdaptiveAvgPool3dOutSize1 instance during forward.
            kwargs (any): any keyword argument (unused).
        """
        assert self.convert_flag is False, 'AdaptiveAvgPool3dOutSize1: already converted, cannot be converted again'
        kernel_size = input_blob_size[2:]
        self.pool = nn.AvgPool3d(kernel_size)
        self.convert_flag = True

    def forward(self, x):
        return self.pool(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
