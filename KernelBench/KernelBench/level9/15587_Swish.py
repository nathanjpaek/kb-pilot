import torch


def swish(tensor: 'torch.Tensor', beta: 'float'=1.0) ->torch.Tensor:
    """
    Applies Swish function element-wise.

    See :class:`torchlayers.activations.Swish` for more details.

    Arguments:
        tensor :
            Tensor activated element-wise
        beta :
            Multiplier used for sigmoid. Default: 1.0 (no multiplier)

    Returns:
        torch.Tensor:
    """
    return torch.sigmoid(beta * tensor) * tensor


class Swish(torch.nn.Module):
    """
    Applies Swish function element-wise.

    !!!math

        Swish(x) = x / (1 + \\exp(-beta * x))

    This form was originally proposed by Prajit Ramachandran et al. in
    `Searching for Activation Functions <https://arxiv.org/pdf/1710.05941.pdf>`__

    Attributes:
        beta :
            Multiplier used for sigmoid. Default: 1.0 (no multiplier)

    """

    def __init__(self, beta: 'float'=1.0):
        """Initialize `Swish` object.
        
        Arguments:
            beta :
                Multiplier used for sigmoid. Default: 1.0 (no multiplier)
        """
        super().__init__()
        self.beta = beta

    def forward(self, tensor: 'torch.Tensor'):
        return swish(tensor, self.beta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
