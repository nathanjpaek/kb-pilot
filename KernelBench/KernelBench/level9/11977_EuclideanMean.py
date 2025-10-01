import torch
from torch import Tensor
import torch.utils.data.dataloader
from torch import nn
import torch.nn


class EuclideanMean(nn.Module):
    """Implement a EuclideanMean object."""

    def forward(self, data: 'Tensor') ->Tensor:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor

        """
        return data.mean(0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
