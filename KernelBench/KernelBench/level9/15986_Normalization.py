import numbers
import torch
import torch.nn as nn


class Normalization(nn.Module):
    """A normalization layer."""

    def __init__(self, eps: 'numbers.Real'=1e-15):
        """Creates a new instance of ``Normalization``.
        
        Args:
            eps (numbers.Real, optional): A tiny number to be added to the standard deviation before re-scaling the
                centered values. This prevents divide-by-0 errors. By default, this is set to ``1e-15``.
        """
        super().__init__()
        self._eps = None
        self.eps = float(eps)

    @property
    def eps(self) ->float:
        """float: A tiny number that is added to the standard deviation before re-scaling the centered values.
        
        This prevents divide-by-0 errors. By default, this is set to ``1e-15``.
        """
        return self._eps

    @eps.setter
    def eps(self, eps: 'numbers.Real') ->None:
        if not isinstance(eps, numbers.Real):
            raise TypeError('<eps> has to be a real number!')
        self._eps = float(eps)

    def forward(self, x: 'torch.FloatTensor') ->torch.FloatTensor:
        """Runs the normalization layer.
        
        Args:
            x (torch.FloatTensor): A tensor to be normalized. To that end, ``x`` is interpreted as a batch of values
                where normalization is applied over the last of its dimensions.
        
        Returns:
            torch.FloatTensor: The normalized tensor.
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return (x - mean) / (std + self._eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
