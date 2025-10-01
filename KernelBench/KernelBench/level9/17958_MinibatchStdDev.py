from torch.nn import Module
import torch
from torch import Tensor
from typing import List


class MinibatchStdDev(Module):
    """
    Minibatch standard deviation layer for the discriminator
    Args:
        group_size: Size of each group into which the batch is split
        num_new_features: number of additional feature maps added
    """

    def __init__(self, group_size: 'int'=4, num_new_features: 'int'=1) ->None:
        """

        Args:
            group_size:
            num_new_features:
        """
        super(MinibatchStdDev, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def extra_repr(self) ->str:
        return (
            f'group_size={self.group_size}, num_new_features={self.num_new_features}'
            )

    def forward(self, x: 'Tensor', alpha: 'float'=1e-08) ->Tensor:
        """
        forward pass of the layer
        Args:
            x: input activation volume
            alpha: small number for numerical stability
        Returns: y => x appended with standard deviation constant map
        """
        batch_size, channels, height, width = x.shape
        y = torch.reshape(x, [batch_size, self.num_new_features, channels //
            self.num_new_features, height, width])
        y_split = y.split(self.group_size)
        y_list: 'List[Tensor]' = []
        for y in y_split:
            group_size = y.shape[0]
            y = y - y.mean(dim=0, keepdim=True)
            y = torch.sqrt(y.square().mean(dim=0, keepdim=False) + alpha)
            y = y.mean(dim=[1, 2, 3], keepdim=True)
            y = y.mean(dim=1, keepdim=False)
            y = y.view((1, *y.shape)).repeat(group_size, 1, height, width)
            y_list.append(y)
        y = torch.cat(y_list, dim=0)
        y = torch.cat([x, y], 1)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
