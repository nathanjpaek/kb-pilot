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


class MultinomialNLLLoss(nn.Module):
    """
  Computes the negative log-likelihood of the multinomial distribution.

  .. math::
    \\ell(x, y) = L = - y \\cdot \\log(softmax(x))

  Args:
    reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
        'elementwise_mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'elementwise_mean'
  """

    def __init__(self, reduction='elementwise_mean'):
        super(MultinomialNLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        loss = -target * F.log_softmax(input, dim=1)
        return _reduce(loss, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
