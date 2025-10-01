import torch


class GaussianLoss(torch.nn.Module):
    """
    Gaussian log-likelihood loss. It assumes targets `y` with n rows and d
    columns, but estimates `yhat` with n rows and 2d columns. The columns 0:d
    of `yhat` contain estimated means, the columns d:2*d of `yhat` contain
    estimated variances. This module assumes that the estimated variances are
    positive---for numerical stability, it is recommended that the minimum
    estimated variance is greater than a small number (1e-3).
    """

    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, yhat, y):
        dim = yhat.size(1) // 2
        mean = yhat[:, :dim]
        variance = yhat[:, dim:]
        term1 = variance.log().div(2)
        term2 = (y - mean).pow(2).div(variance.mul(2))
        return (term1 + term2).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 2, 4, 4])]


def get_init_inputs():
    return [[], {}]
