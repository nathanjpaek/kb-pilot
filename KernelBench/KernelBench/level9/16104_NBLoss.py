import torch
import numpy as np


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class NBLoss(torch.nn.Module):

    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, yhat, y, eps=1e-08):
        """Negative binomial log-likelihood loss. It assumes targets `y` with n
        rows and d columns, but estimates `yhat` with n rows and 2d columns.
        The columns 0:d of `yhat` contain estimated means, the columns d:2*d of
        `yhat` contain estimated variances. This module assumes that the
        estimated mean and inverse dispersion are positive---for numerical
        stability, it is recommended that the minimum estimated variance is
        greater than a small number (1e-3).
        Parameters
        ----------
        yhat: Tensor
                Torch Tensor of reeconstructed data.
        y: Tensor
                Torch Tensor of ground truth data.
        eps: Float
                numerical stability constant.
        """
        dim = yhat.size(1) // 2
        mu = yhat[:, :dim]
        theta = yhat[:, dim:]
        if theta.ndimension() == 1:
            theta = theta.view(1, theta.size(0))
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y + 1.0) - torch.lgamma(
            y + theta + eps)
        t2 = (theta + y) * torch.log(1.0 + mu / (theta + eps)) + y * (torch
            .log(theta + eps) - torch.log(mu + eps))
        final = t1 + t2
        final = _nan2inf(final)
        return torch.mean(final)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 2, 4, 4])]


def get_init_inputs():
    return [[], {}]
