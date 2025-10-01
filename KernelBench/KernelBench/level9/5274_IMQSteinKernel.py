import math
import torch


def norm_sq(X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    return -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)


class IMQSteinKernel(torch.nn.Module):
    """
    IMQ (inverse multi-quadratic) kernel

    :math:`K(x, y) = (\\alpha + ||x-y||^2/h)^{\\beta}`

    """

    def __init__(self, alpha=0.5, beta=-0.5, bandwidth=None):
        super(IMQSteinKernel, self).__init__()
        assert alpha > 0.0, 'alpha must be positive.'
        assert beta < 0.0, 'beta must be negative.'
        self.alpha = alpha
        self.beta = beta
        self.bandwidth = bandwidth

    def _bandwidth(self, norm_sq):
        """
        Compute the bandwidth along each dimension using the median pairwise squared distance between particles.
        """
        if self.bandwidth is None:
            num_particles = norm_sq.size(0)
            index = torch.arange(num_particles)
            norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
            median = norm_sq.median(dim=0)[0]
            assert median.shape == norm_sq.shape[-1:]
            return median / math.log(num_particles + 1)
        else:
            return self.bandwidth

    def forward(self, X, Y):
        norm_sq = (X.unsqueeze(0) - Y.unsqueeze(1)) ** 2
        assert norm_sq.dim() == 3
        bandwidth = self._bandwidth(norm_sq)
        base_term = self.alpha + torch.sum(norm_sq / bandwidth, dim=-1)
        log_kernel = self.beta * torch.log(base_term)
        return log_kernel.exp()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
