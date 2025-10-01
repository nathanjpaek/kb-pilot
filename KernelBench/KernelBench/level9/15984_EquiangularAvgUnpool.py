import math
import torch
from torch import nn
import torch.nn.functional as F


def equiangular_bandwidth(nodes):
    """Calculate the equiangular bandwidth based on input nodes

    Args:
        nodes (int): the number of nodes should be a power of 4

    Returns:
        int: the corresponding bandwidth
    """
    bw = math.sqrt(nodes) / 2
    return bw


def equiangular_dimension_unpack(nodes, ratio):
    """Calculate the two underlying dimensions
    from the total number of nodes

    Args:
        nodes (int): combined dimensions
        ratio (float): ratio between the two dimensions

    Returns:
        int, int: separated dimensions
    """
    dim1 = int((nodes / ratio) ** 0.5)
    dim2 = int((nodes * ratio) ** 0.5)
    return dim1, dim2


def equiangular_calculator(tensor, ratio):
    """From a 3D input tensor and a known ratio between the latitude
    dimension and longitude dimension of the data, reformat the 3D input
    into a 4D output while also obtaining the bandwidth.

    Args:
        tensor (:obj:`torch.tensor`): 3D input tensor
        ratio (float): the ratio between the latitude and longitude dimension of the data

    Returns:
        :obj:`torch.tensor`, int, int: 4D tensor, the bandwidths for lat. and long.
    """
    N, M, F = tensor.size()
    dim1, dim2 = equiangular_dimension_unpack(M, ratio)
    bw_dim1 = equiangular_bandwidth(dim1)
    bw_dim2 = equiangular_bandwidth(dim2)
    tensor = tensor.view(N, dim1, dim2, F)
    return tensor, [bw_dim1, bw_dim2]


def reformat(x):
    """Reformat the input from a 4D tensor to a 3D tensor

    Args:
        x (:obj:`torch.tensor`): a 4D tensor
    Returns:
        :obj:`torch.tensor`: a 3D tensor
    """
    x = x.permute(0, 2, 3, 1)
    N, D1, D2, Feat = x.size()
    x = x.view(N, D1 * D2, Feat)
    return x


class EquiangularAvgUnpool(nn.Module):
    """EquiAngular Average Unpooling version 1 using the interpolate function when unpooling
    """

    def __init__(self, ratio):
        """Initialization

        Args:
            ratio (float): ratio between latitude and longitude dimensions of the data
        """
        self.ratio = ratio
        self.kernel_size = 4
        super().__init__()

    def forward(self, x):
        """calls pytorch's interpolate function to create the values while unpooling based on the nearby values
        Args:
            x (:obj:`torch.tensor`): batch x pixels x features
        Returns:
            :obj:`torch.tensor`: batch x unpooled pixels x features
        """
        x, _ = equiangular_calculator(x, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=(self.kernel_size, self.
            kernel_size), mode='nearest')
        x = reformat(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'ratio': 4}]
