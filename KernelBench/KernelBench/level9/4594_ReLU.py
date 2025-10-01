import torch
import torch.nn as nn
from abc import abstractmethod
import torch.utils.data
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


class ReLU(EfficientBlockBase):
    """
    ReLU activation function for EfficientBlockBase.
    """

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x)

    def convert(self, *args, **kwarg):
        pass


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
