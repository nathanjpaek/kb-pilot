import torch
import numpy as np
import torch.nn as nn


class ShiftedSoftplus(nn.Module):
    """ Shifted softplus (SSP) activation function

    SSP originates from the softplus function:

        y = \\ln\\left(1 + e^{-x}\\right)

    Schütt et al. (2018) introduced a shifting factor to the function in order
    to ensure that SSP(0) = 0 while having infinite order of continuity:

         y = \\ln\\left(1 + e^{-x}\\right) - \\ln(2)

    SSP allows to obtain smooth potential energy surfaces and second derivatives
    that are required for training with forces as well as the calculation of
    vibrational modes (Schütt et al. 2018).

    References
    ----------
    K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela,
        A. Tkatchenko, K.-R. Müller. (2018)
        SchNet - a deep learning architecture for molecules and materials.
        The Journal of Chemical Physics.
        https://doi.org/10.1063/1.5019779

    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, input_tensor):
        """ Applies the shifted softplus function element-wise

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of size (n_examples, *) where `*` means, any number of
            additional dimensions.

        Returns
        -------
        Output: torch.Tensor
            Same size (n_examples, *) as the input.
        """
        return nn.functional.softplus(input_tensor) - np.log(2.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
