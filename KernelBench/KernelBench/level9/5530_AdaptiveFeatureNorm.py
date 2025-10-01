import torch
import torch.nn as nn
import torch.utils.data


class AdaptiveFeatureNorm(nn.Module):
    """
    The `Stepwise Adaptive Feature Norm loss (ICCV 2019) <https://arxiv.org/pdf/1811.07456v2.pdf>`_

    Instead of using restrictive scalar R to match the corresponding feature norm, Stepwise Adaptive Feature Norm
    is used in order to learn task-specific features with large norms in a progressive manner.
    Given feature representations :math:`f` on source or target domain, the definition of Stepwise Adaptive Feature Norm loss is

    .. math::
        norm\\_loss = \\mathbb{E}_{i}(\\Vert f_i \\Vert_2.detach() + delta - \\Vert f_i \\Vert_2)^2\\\\

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
