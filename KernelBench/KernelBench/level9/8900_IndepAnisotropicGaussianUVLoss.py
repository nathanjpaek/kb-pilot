import math
import torch
from torch.nn import functional as F
import torch.utils.data
from torch import nn


class IndepAnisotropicGaussianUVLoss(nn.Module):
    """
    Loss for the case of independent residuals with anisotropic covariances:
    $Sigma_i = sigma_i^2 I + r_i r_i^T$
    The loss (negative log likelihood) is then:
    $1/2 sum_{i=1}^n (log(2 pi)
      + log sigma_i^2 (sigma_i^2 + ||r_i||^2)
      + ||delta_i||^2 / sigma_i^2
      - <delta_i, r_i>^2 / (sigma_i^2 * (sigma_i^2 + ||r_i||^2)))$,
    where $delta_i=(u - u', v - v')$ is a 2D vector containing UV coordinates
    difference between estimated and ground truth UV values
    For details, see:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
    Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    """

    def __init__(self, sigma_lower_bound: 'float'):
        super(IndepAnisotropicGaussianUVLoss, self).__init__()
        self.sigma_lower_bound = sigma_lower_bound
        self.log2pi = math.log(2 * math.pi)

    def forward(self, u: 'torch.Tensor', v: 'torch.Tensor', sigma_u:
        'torch.Tensor', kappa_u_est: 'torch.Tensor', kappa_v_est:
        'torch.Tensor', target_u: 'torch.Tensor', target_v: 'torch.Tensor'):
        sigma2 = F.softplus(sigma_u) + self.sigma_lower_bound
        r_sqnorm2 = kappa_u_est ** 2 + kappa_v_est ** 2
        delta_u = u - target_u
        delta_v = v - target_v
        delta_sqnorm = delta_u ** 2 + delta_v ** 2
        delta_u_r_u = delta_u * kappa_u_est
        delta_v_r_v = delta_v * kappa_v_est
        delta_r = delta_u_r_u + delta_v_r_v
        delta_r_sqnorm = delta_r ** 2
        denom2 = sigma2 * (sigma2 + r_sqnorm2)
        loss = 0.5 * (self.log2pi + torch.log(denom2) + delta_sqnorm /
            sigma2 - delta_r_sqnorm / denom2)
        return loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]),
        torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sigma_lower_bound': 4}]
