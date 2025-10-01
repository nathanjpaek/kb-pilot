import torch
from torch import nn


class LabelSmoothingCrossEntropyV2(nn.Module):
    """ 平滑的交叉熵, LabelSommth-CrossEntropy
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    url: https://github.com/CoinCheung/pytorch-loss
    examples:
        >>> criteria = LabelSmoothingCrossEntropyV2()
        >>> logits = torch.randn(8, 19, 384, 384)  # nchw, float/half
        >>> lbs = torch.randint(0, 19, (8, 384, 384))  # nhw, int64_t
        >>> loss = criteria(logits, lbs)
    """

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropyV2, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.lb_ignore = ignore_index
        self.lb_smooth = lb_smooth
        self.reduction = reduction

    def forward(self, logits, label):
        logits = logits.float()
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1.0 - self.lb_smooth, self.lb_smooth / num_classes
            label_unsq = label.unsqueeze(1)
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1,
                label_unsq, lb_pos).detach()
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=
        torch.int64)]


def get_init_inputs():
    return [[], {}]
