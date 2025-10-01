import torch
import torch.nn as nn
import torch.utils.data.distributed
from torch.backends import cudnn as cudnn


class SoftJaccardLoss(nn.Module):
    """SoftJaccard loss for binary problems.
    """

    def __init__(self, use_log=False):
        super(SoftJaccardLoss, self).__init__()
        self.use_log = use_log

    def forward(self, logits, labels):
        num = labels.size(0)
        m1 = torch.sigmoid(logits.view(num, -1))
        m2 = labels.view(num, -1)
        intersection = (m1 * m2).sum(1)
        score = (intersection + 1e-15) / (m1.sum(1) + m2.sum(1) -
            intersection + 1e-15)
        jaccard = score.sum(0) / num
        if not self.use_log:
            score = 1 - jaccard
        else:
            score = -torch.log(jaccard)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
