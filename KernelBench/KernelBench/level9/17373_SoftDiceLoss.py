import torch
import torch.nn as nn
import torch.utils.data.distributed
from torch.backends import cudnn as cudnn


class SoftDiceLoss(nn.Module):
    """SoftJaccard loss for binary problems.
    """

    def forward(self, logits, labels):
        num = labels.size(0)
        m1 = torch.sigmoid(logits.view(num, -1))
        m2 = labels.view(num, -1)
        intersection = (m1 * m2).sum(1)
        score = (intersection + 1e-15) / (m1.sum(1) + m2.sum(1) + 1e-15)
        dice = score.sum(0) / num
        return 1 - dice


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
