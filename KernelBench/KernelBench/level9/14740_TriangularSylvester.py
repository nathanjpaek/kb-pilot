import torch
from torch import nn


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):
        super(TriangularSylvester, self).__init__()
        self.z_size = z_size
        self.h = nn.Tanh()

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """
        zk = zk.unsqueeze(1)
        diag_r1 = torch.diagonal(r1, 0, -1, -2)
        diag_r2 = torch.diagonal(r2, 0, -1, -2)
        if permute_z is not None:
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk
        r2qzb = z_per @ r2.transpose(2, 1) + b
        z = self.h(r2qzb) @ r1.transpose(2, 1)
        if permute_z is not None:
            z = z[:, :, permute_z]
        z += zk
        z = z.squeeze(1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.0
        log_diag_j = (diag_j.abs() + 1e-08).log()
        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j
        return z, log_det_j


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_size': 4}]
