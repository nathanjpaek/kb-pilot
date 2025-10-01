import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed


class AdaptiveFeatureNorm(nn.Module):
    """
    The `Stepwise Adaptive Feature Norm loss (ICCV 2019) <https://arxiv.org/pdf/1811.07456v2.pdf>`_

    Instead of using restrictive scalar R to match the corresponding feature norm, Stepwise Adaptive Feature Norm
    is used in order to learn task-specific features with large norms in a progressive manner.
    We denote parameters of backbone :math:`G` as :math:`\\theta_g`, parameters of bottleneck :math:`F_f` as :math:`\\theta_f`
    , parameters of classifier head :math:`F_y` as :math:`\\theta_y`, and features extracted from sample :math:`x_i` as
    :math:`h(x_i;\\theta)`. Full loss is calculated as follows

    .. math::
        L(\\theta_g,\\theta_f,\\theta_y)=\\frac{1}{n_s}\\sum_{(x_i,y_i)\\in D_s}L_y(x_i,y_i)+\\frac{\\lambda}{n_s+n_t}
        \\sum_{x_i\\in D_s\\cup D_t}L_d(h(x_i;\\theta_0)+\\Delta_r,h(x_i;\\theta))\\\\

    where :math:`L_y` denotes classification loss, :math:`L_d` denotes norm loss, :math:`\\theta_0` and :math:`\\theta`
    represent the updated and updating model parameters in the last and current iterations respectively.

    Args:
        delta (float): positive residual scalar to control the feature norm enlargement.

    Inputs:
        - f (tensor): feature representations on source or target domain.

    Shape:
        - f: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    Examples::

        >>> adaptive_feature_norm = AdaptiveFeatureNorm(delta=1)
        >>> f_s = torch.randn(32, 1000)
        >>> f_t = torch.randn(32, 1000)
        >>> norm_loss = adaptive_feature_norm(f_s) + adaptive_feature_norm(f_t)
    """

    def __init__(self, delta):
        super(AdaptiveFeatureNorm, self).__init__()
        self.delta = delta

    def forward(self, f: 'torch.Tensor') ->torch.Tensor:
        radius = f.norm(p=2, dim=1).detach()
        assert radius.requires_grad is False
        radius = radius + self.delta
        loss = ((f.norm(p=2, dim=1) - radius) ** 2).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'delta': 4}]
