import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data


def _is_long(x):
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.LongTensor)


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def smoothing(out, y, smooth_eps):
    num_classes = out.shape[1]
    if smooth_eps == 0:
        return y
    my = onehot(y, num_classes)
    true_class, false_class = 1.0 - smooth_eps * num_classes / (num_classes - 1
        ), smooth_eps / (num_classes - 1)
    my = my * true_class + torch.ones_like(my) * false_class
    return my


def cross_entropy(logits, target, weight=None, ignore_index=-100, reduction
    ='mean', smooth_eps=0.0):
    """cross entropy loss with support for target distributions"""
    with torch.no_grad():
        if smooth_eps > 0:
            target = smoothing(logits, target, smooth_eps)
    if _is_long(target):
        return F.cross_entropy(logits, target, weight, ignore_index=
            ignore_index, reduction=reduction)
    masked_indices = None
    logits.size(-1)
    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)
    lsm = F.log_softmax(logits, dim=-1)
    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)
    loss = -(target * lsm).sum(-1)
    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())
    return loss


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets and built-in label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean',
        smooth_eps=0.0):
        super(CrossEntropyLoss, self).__init__(weight=weight, ignore_index=
            ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        return cross_entropy(input, target, self.weight, self.ignore_index,
            self.reduction, self.smooth_eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
