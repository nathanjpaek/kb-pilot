import math
import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class IIDIsotropicGaussianUVLoss(nn.Module):
    """
    Loss for the case of iid residuals with isotropic covariance:
    $Sigma_i = sigma_i^2 I$
    The loss (negative log likelihood) is then:
    $1/2 sum_{i=1}^n (log(2 pi) + 2 log sigma_i^2 + ||delta_i||^2 / sigma_i^2)$,
    where $delta_i=(u - u', v - v')$ is a 2D vector containing UV coordinates
    difference between estimated and ground truth UV values
    For details, see:
    N. Neverova, D. Novotny, A. Vedaldi "Correlated Uncertainty for Learning
    Dense Correspondences from Noisy Labels", p. 918--926, in Proc. NIPS 2019
    """

    def __init__(self, sigma_lower_bound: 'float'):
        super(IIDIsotropicGaussianUVLoss, self).__init__()
        self.sigma_lower_bound = sigma_lower_bound
        self.log2pi = math.log(2 * math.pi)

    def forward(self, u: 'torch.Tensor', v: 'torch.Tensor', sigma_u:
        'torch.Tensor', target_u: 'torch.Tensor', target_v: 'torch.Tensor'):
        sigma2 = F.softplus(sigma_u) + self.sigma_lower_bound
        delta_t_delta = (u - target_u) ** 2 + (v - target_v) ** 2
        loss = 0.5 * (self.log2pi + 2 * torch.log(sigma2) + delta_t_delta /
            sigma2)
        return loss.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'sigma_lower_bound': 4}]
