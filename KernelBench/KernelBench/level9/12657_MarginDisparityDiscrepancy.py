import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.optim


def shift_log(x: 'torch.Tensor', offset: 'Optional[float]'=1e-06
    ) ->torch.Tensor:
    """
    First shift, then calculate log, which can be described as:

    .. math::
        y = \\max(\\log(x+\\text{offset}), 0)

    Used to avoid the gradient explosion problem in log(x) function when x=0.

    Parameters:
        - **x**: input tensor
        - **offset**: offset size. Default: 1e-6

    .. note::
        Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
    """
    return torch.log(torch.clamp(x + offset, max=1.0))


class MarginDisparityDiscrepancy(nn.Module):
    """The margin disparity discrepancy (MDD) is proposed to measure the distribution discrepancy in domain adaptation.

    The :math:`y^s` and :math:`y^t` are logits output by the main classifier on the source and target domain respectively.
    The :math:`y_{adv}^s` and :math:`y_{adv}^t` are logits output by the adversarial classifier.
    They are expected to contain raw, unnormalized scores for each class.

    The definition can be described as:

    .. math::
        \\mathcal{D}_{\\gamma}(\\hat{\\mathcal{S}}, \\hat{\\mathcal{T}}) =
        \\gamma \\mathbb{E}_{y^s, y_{adv}^s \\sim\\hat{\\mathcal{S}}} \\log\\left(\\frac{\\exp(y_{adv}^s[h_{y^s}])}{\\sum_j \\exp(y_{adv}^s[j])}\\right) +
        \\mathbb{E}_{y^t, y_{adv}^t \\sim\\hat{\\mathcal{T}}} \\log\\left(1-\\frac{\\exp(y_{adv}^t[h_{y^t}])}{\\sum_j \\exp(y_{adv}^t[j])}\\right),

    where :math:`\\gamma` is a margin hyper-parameter and :math:`h_y` refers to the predicted label when the logits output is :math:`y`.
    You can see more details in `Bridging Theory and Algorithm for Domain Adaptation <https://arxiv.org/abs/1904.05801>`_.

    Parameters:
        - **margin** (float): margin :math:`\\gamma`. Default: 4
        - **reduction** (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs: y_s, y_s_adv, y_t, y_t_adv
        - **y_s**: logits output :math:`y^s` by the main classifier on the source domain
        - **y_s_adv**: logits output :math:`y^s` by the adversarial classifier on the source domain
        - **y_t**: logits output :math:`y^t` by the main classifier on the target domain
        - **y_t_adv**: logits output :math:`y_{adv}^t` by the adversarial classifier on the target domain

    Shape:
        - Inputs: :math:`(minibatch, C)` where C = number of classes, or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
          with :math:`K \\geq 1` in the case of `K`-dimensional loss.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(minibatch)`, or
          :math:`(minibatch, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of K-dimensional loss.

    Examples::
        >>> num_classes = 2
        >>> batch_size = 10
        >>> loss = MarginDisparityDiscrepancy(margin=4.)
        >>> # logits output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> # adversarial logits output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(y_s, y_s_adv, y_t, y_t_adv)
    """

    def __init__(self, margin: 'Optional[int]'=4, reduction:
        'Optional[str]'='mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, y_s: 'torch.Tensor', y_s_adv: 'torch.Tensor', y_t:
        'torch.Tensor', y_t_adv: 'torch.Tensor') ->torch.Tensor:
        _, prediction_s = y_s.max(dim=1)
        _, prediction_t = y_t.max(dim=1)
        return self.margin * F.cross_entropy(y_s_adv, prediction_s,
            reduction=self.reduction) + F.nll_loss(shift_log(1.0 - F.
            softmax(y_t_adv, dim=1)), prediction_t, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
