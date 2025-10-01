import torch
from torch import nn


class LabelSmoothingCrossEntropyV1(nn.Module):

    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        """【直接smooth输入logits效果不好】LabelSmoothingCrossEntropy, no-softmax-input
        eps==0-1, 通过控制ce权重、新增后置项来处理来平滑
        urls: [pytorch | labelSmooth](https://zhuanlan.zhihu.com/p/265704145)
        args:
            ignore_index: (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient. Default: -100
            reduction: str, Specifies the reduction to apply to the output, 输出形式. 
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            eps: float, Minimum of maths, 极小值.  eg. 0.1
        returns:
            Tensor of loss.
        examples:
        >>> loss = LabelSmoothingCrossEntropyV1()(logits, label)
        """
        super(LabelSmoothingCrossEntropyV1, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, labels):
        V = max(logits.size()[-1] - 1, 1)
        logits_smooth = (1 - self.eps) * logits + self.eps / V
        logits_smooth_logsigmoid = torch.nn.functional.logsigmoid(logits_smooth
            )
        loss = -(labels * logits_smooth_logsigmoid + (1 - labels) *
            logits_smooth_logsigmoid)
        loss = loss.sum(dim=1)
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
