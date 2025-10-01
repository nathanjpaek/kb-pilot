import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.onnx


def _reduce(x, reduction='elementwise_mean'):
    if reduction == 'none':
        return x
    elif reduction == 'elementwise_mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()
    else:
        raise ValueError('No such reduction {} defined'.format(reduction))


class MSELoss(nn.Module):
    """
  Computes the weighted mean squared error loss.

  The weight for an observation x:

  .. math::
    w = 1 + confidence \\times x

  and the loss is:

  .. math::
    \\ell(x, y) = w \\cdot (y - x)^2

  Args:
    confidence (float, optional): the weighting of positive observations.
    reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'elementwise_mean'
  """

    def __init__(self, confidence=0, reduction='elementwise_mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        self.confidence = confidence

    def forward(self, input, target):
        weights = 1 + self.confidence * (target > 0).float()
        loss = F.mse_loss(input, target, reduction='none')
        weighted_loss = weights * loss
        return _reduce(weighted_loss, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
