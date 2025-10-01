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


class SH2Signal(nn.Module):
    """
    SH2Signal(dwi_sh) -> dwi

    Computes the corresponding dwi signal for each gradient

    Args:
        x_in (5D tensor): input spherical harmonic tensor
        x_in.size(): (Batchsize x Number of shells*Number of coefficients x DimX x DimY x DimZ)
        y (5D tensor): corresponding dwi tensor
        y.size(): (Batchsize x Number of shells * Number of gradients x DimX x DimY x DimZ)
    """

    def __init__(self, sh_order, gradients):
        super(SH2Signal, self).__init__()
        self.sh_order = sh_order
        self.num_gradients = gradients.shape[0]
        self.num_coefficients = int((self.sh_order + 1) * (self.sh_order / 
            2 + 1))
        SH2SignalMat = np.zeros((self.num_coefficients, self.num_gradients))
        for id_gradient in range(self.num_gradients):
            id_coefficient = 0
            for id_order in range(0, self.sh_order + 1, 2):
                for id_degree in range(-id_order, id_order + 1):
                    gradients_phi, gradients_theta, _gradients_z = cart2sph(
                        gradients[id_gradient, 0], gradients[id_gradient, 1
                        ], gradients[id_gradient, 2])
                    y = sci.sph_harm(np.abs(id_degree), id_order,
                        gradients_phi, gradients_theta)
                    if id_degree < 0:
                        SH2SignalMat[id_coefficient, id_gradient] = np.real(y
                            ) * np.sqrt(2)
                    elif id_degree == 0:
                        SH2SignalMat[id_coefficient, id_gradient] = np.real(y)
                    elif id_degree > 0:
                        SH2SignalMat[id_coefficient, id_gradient] = np.imag(y
                            ) * np.sqrt(2)
                    id_coefficient += 1
        self.SH2SignalMat = torch.nn.Parameter(torch.from_numpy(
            SH2SignalMat).float(), requires_grad=False)

    def forward(self, x_in):
        x_dim = x_in.size()
        x = x_in.reshape((x_dim[0], np.ceil(x_in.size(1) / self.
            num_coefficients).astype(int), self.num_coefficients, x_dim[-3],
            x_dim[-2], x_dim[-1]))
        x = x.permute(0, 1, 3, 4, 5, 2)
        y = x.matmul(self.SH2SignalMat)
        y = y.permute(0, 1, 5, 2, 3, 4).contiguous().reshape((x_dim[0], -1,
            x_dim[-3], x_dim[-2], x_dim[-1]))
        return y


def get_inputs():
    return [torch.rand([4, 1, 15, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sh_order': 4, 'gradients': torch.rand([4, 4])}]
