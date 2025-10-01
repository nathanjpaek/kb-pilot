import math
import torch


class Tanh(torch.nn.Tanh):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def forward(self, inputs, grad: 'torch.Tensor'=None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous 
        transformations combined with this transformation.
        """
        g = -2 * (inputs - math.log(2) + torch.nn.functional.softplus(-2 *
            inputs))
        return torch.tanh(inputs), g.view(grad.shape
            ) + grad if grad is not None else g


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
