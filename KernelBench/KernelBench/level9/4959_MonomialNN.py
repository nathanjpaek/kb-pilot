import torch
import torch.nn as nn
from warnings import warn


class MonomialNN(nn.Module):
    """A network that expands its input to a given list of monomials.

    Its output shape will be (n_samples, n_input_units * n_degrees)

    :param degrees: max degree to be included, or a list of degrees that will be used
    :type degrees: int or list[int] or tuple[int]
    """

    def __init__(self, degrees):
        super(MonomialNN, self).__init__()
        if isinstance(degrees, int):
            degrees = [d for d in range(1, degrees + 1)]
        self.degrees = tuple(degrees)
        if len(self.degrees) == 0:
            raise ValueError('No degrees used, check `degrees` argument again')
        if 0 in degrees:
            warn(
                'One of the degrees is 0 which might introduce redundant features'
                )
        if len(set(self.degrees)) < len(self.degrees):
            warn(f'Duplicate degrees found: {self.degrees}')

    def forward(self, x):
        return torch.cat([(x ** d) for d in self.degrees], dim=1)

    def __repr__(self):
        return f'{self.__class__.__name__}(degrees={self.degrees})'

    def __str__(self):
        return self.__repr__()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'degrees': 4}]
