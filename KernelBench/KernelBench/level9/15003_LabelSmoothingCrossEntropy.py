import torch
from torch import nn


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        """LabelSmoothingCrossEntropy, no-softmax-input
        对logits进行smoothing, 即log_softmax后进行操作
        args:
            ignore_index: (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: -100
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            eps: float, Minimum of maths, 极小值.  eg. 0.1
        returns:
            Tensor of loss.
        examples:
          >>> loss = LabelSmoothingCrossEntropy()(logits, label)
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, labels):
        V = max(logits.size()[-1] - 1, 1)
        loss = (1 - self.eps) * -(labels * torch.nn.functional.logsigmoid(
            logits) + (1 - labels) * torch.nn.functional.logsigmoid(-logits)
            ) + self.eps / V
        loss = loss.sum(dim=1) / logits.size(1)
        if 'mean' == self.reduction:
            loss = loss.mean()
        elif 'sum' == self.reduction:
            loss = loss.sum()
        else:
            _
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
