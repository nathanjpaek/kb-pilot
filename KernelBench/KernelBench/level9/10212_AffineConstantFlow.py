import torch
from torch import Tensor
from torch import nn


class FlowBlock(nn.Module):
    """
    Abstract base class for any flow blocks.
    """

    def __init__(self, dimension):
        super(FlowBlock, self).__init__()
        self.dimension = dimension

    def forward(self, x: 'Tensor') ->(Tensor, Tensor):
        """
        When implemented, forward method will represent z = f(x) and log |det f'(x)/dx|
        x: (*, dimension), z: (*, dimension) and log_det: (*, 1)
        """
        raise NotImplementedError('Forward not implemented')

    def inverse(self, z: 'Tensor') ->(Tensor, Tensor):
        """
        When implemented, inverse method will represent x = f^-(z) and log |det f^-'(z)/dz|
        z: (*, dimension), x: (*, dimension) and log_det: (*, 1)
        """
        raise NotImplementedError('Inverse not implemented')


class AffineConstantFlow(FlowBlock):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dimension, scale=True, shift=True):
        super().__init__(dimension)
        zeros = torch.zeros(size=(1, dimension))
        self.s = nn.Parameter(torch.randn(1, dimension, requires_grad=True)
            ) if scale else zeros
        self.t = nn.Parameter(torch.randn(1, dimension, requires_grad=True)
            ) if shift else zeros

    def forward(self, x) ->(Tensor, Tensor):
        z = x * torch.exp(self.s) + self.t
        log_det = torch.sum(self.s, dim=1)
        return z, log_det.repeat(x.shape[0], 1)

    def inverse(self, z) ->(Tensor, Tensor):
        x = (z - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s, dim=1)
        return x, log_det.repeat(z.shape[0], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dimension': 4}]
