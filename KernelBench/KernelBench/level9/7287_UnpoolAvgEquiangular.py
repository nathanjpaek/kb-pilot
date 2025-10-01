import torch
from torch.nn import functional as F


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
    if dim1 * dim2 != nodes:
        if nodes % dim1 == 0:
            dim2 = nodes // dim1
        if nodes % dim2 == 0:
            dim1 = nodes // dim2
    assert dim1 * dim2 == nodes, f'Unable to unpack nodes: {nodes}, ratio: {ratio}'
    return dim1, dim2


def equiangular_calculator(tensor, ratio):
    N, M, F = tensor.size()
    dim1, dim2 = equiangular_dimension_unpack(M, ratio)
    tensor = tensor.view(N, dim1, dim2, F)
    return tensor


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


class UnpoolAvgEquiangular(torch.nn.Module):
    """EquiAngular average unpooling
    
    Parameters
    ----------
    ratio : float
        Parameter for equiangular sampling -> width/height
    """

    def __init__(self, ratio, kernel_size, *args, **kwargs):
        self.ratio = ratio
        self.kernel_size = int(kernel_size ** 0.5)
        super().__init__()

    def forward(self, inputs, *args):
        """calls pytorch's interpolate function to create the values while unpooling based on the nearby values
        Parameters
        ----------
        inputs : torch.tensor of shape batch x pixels x features
            Input data
        
        Returns
        -------
        x : torch.tensor of shape batch x unpooled pixels x features
            Layer output
        """
        x = equiangular_calculator(inputs, self.ratio)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, scale_factor=(self.kernel_size, self.
            kernel_size), mode='nearest')
        x = reformat(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'ratio': 4, 'kernel_size': 4}]
