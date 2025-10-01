import torch
import numpy as np


def norm_sq(X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    return -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)


class RBF_Kernel(torch.nn.Module):
    """
      RBF kernel

      :math:`K(x, y) = exp(||x-v||^2 / (2h))

      """

    def __init__(self, bandwidth=None):
        super().__init__()
        self.bandwidth = bandwidth

    def _bandwidth(self, norm_sq):
        if self.bandwidth is None:
            np_dnorm2 = norm_sq.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(np_dnorm2.shape[0] + 1))
            return np.sqrt(h).item()
        else:
            return self.bandwidth

    def forward(self, X, Y):
        dnorm2 = norm_sq(X, Y)
        bandwidth = self._bandwidth(dnorm2)
        gamma = 1.0 / (1e-08 + 2 * bandwidth ** 2)
        K_XY = (-gamma * dnorm2).exp()
        return K_XY


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
