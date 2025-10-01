from torch.nn import Module
import torch
from torch import Tensor
import torch.optim


class Mish(Module):
    """
        Mish Activation Layer

        Applies a Mish activation function to the input

        Inherits from:
            Module (nn.module.Module)
    """

    def __init__(self) ->None:
        super().__init__()

    def forward(self, x: 'Tensor') ->Tensor:
        """
            Args:
                x (Tensor): (batch_size, num_features)
            
            Returns:
                Tensor: (batch_size, num_features)
        """
        return x * (1 + x.exp()).log().tanh()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
