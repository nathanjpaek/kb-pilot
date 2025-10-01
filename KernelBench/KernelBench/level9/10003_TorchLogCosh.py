import torch
import torch as _torch


class TorchLogCosh(_torch.nn.Module):
    """
    Log(cosh) activation function for PyTorch modules
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return -input + _torch.nn.functional.softplus(2.0 * input)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
