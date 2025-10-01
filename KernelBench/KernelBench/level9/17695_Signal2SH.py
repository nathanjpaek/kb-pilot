import torch
import numpy as np
import torch.nn as nn
from scipy import special as sci


def cart2sph(x, y, z):
    """
    cart2sph(x, y, z) -> theta, phi, r

    Computes the corresponding spherical coordinate of the given input parameters :attr:`x`, :attr:`y` and :attr:`x`.

    Args:
        x (Number): x position
        y (Number): y position
        z (Number): z position

    Example::

        >>> cart2sph(1, 1, 1)
        (0.78539816339744828, 0.95531661812450919, 1.7320508075688772)
    """
    azimuthal_angle = np.arctan2(y, x)
    radial_distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    polar_angle = np.arccos(z / radial_distance)
    return azimuthal_angle, polar_angle, radial_distance


class Signal2SH(nn.Module):
    """
    Signal2SH(dwi) -> dwi_sh

    Computes the corresponding spherical harmonic coefficients

    Args:
        x_in (5D tensor): input dwi tensor
        x_in.size(): (Batchsize x Number of shells * Number of gradients x DimX x DimY x DimZ)
        y (5D tensor): corresponding harmonic coefficients tensor
        y.size(): (Batchsize x Number of shells*Number of coefficients x DimX x DimY x DimZ)
    """

    def __init__(self, sh_order, gradients, lb_lambda=0.006):
        super(Signal2SH, self).__init__()
        self.sh_order = sh_order
        self.lb_lambda = lb_lambda
        self.num_gradients = gradients.shape[0]
        self.num_coefficients = int((self.sh_order + 1) * (self.sh_order / 
            2 + 1))
        b = np.zeros((self.num_gradients, self.num_coefficients))
        l = np.zeros((self.num_coefficients, self.num_coefficients))
        for id_gradient in range(self.num_gradients):
            id_column = 0
            for id_order in range(0, self.sh_order + 1, 2):
                for id_degree in range(-id_order, id_order + 1):
                    gradients_phi, gradients_theta, _gradients_z = cart2sph(
                        gradients[id_gradient, 0], gradients[id_gradient, 1
                        ], gradients[id_gradient, 2])
                    y = sci.sph_harm(np.abs(id_degree), id_order,
                        gradients_phi, gradients_theta)
                    if id_degree < 0:
                        b[id_gradient, id_column] = np.real(y) * np.sqrt(2)
                    elif id_degree == 0:
                        b[id_gradient, id_column] = np.real(y)
                    elif id_degree > 0:
                        b[id_gradient, id_column] = np.imag(y) * np.sqrt(2)
                    l[id_column, id_column
                        ] = self.lb_lambda * id_order ** 2 * (id_order + 1
                        ) ** 2
                    id_column += 1
        b_inv = np.linalg.pinv(np.matmul(b.transpose(), b) + l)
        self.Signal2SHMat = torch.nn.Parameter(torch.from_numpy(np.matmul(
            b_inv, b.transpose()).transpose()).float(), requires_grad=False)

    def forward(self, x_in):
        x = x_in.reshape((-1, np.ceil(x_in.size(1) / self.num_gradients).
            astype(int), self.num_gradients, x_in.size(2), x_in.size(3),
            x_in.size(4)))
        x = x.permute(0, 1, 3, 4, 5, 2)
        y = x.matmul(self.Signal2SHMat)
        y = y.permute(0, 1, 5, 2, 3, 4).contiguous().reshape((x.size(0), -1,
            x_in.size(2), x_in.size(3), x_in.size(4)))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sh_order': 4, 'gradients': torch.rand([4, 4])}]
