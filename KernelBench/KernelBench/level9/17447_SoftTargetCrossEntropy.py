import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, reduce='mean'):
        super(SoftTargetCrossEntropy, self).__init__()
        self.criterion = nn.KLDivLoss(reduction=reduce)
        self.reduce = reduce

    def forward(self, x, target, mask=None):
        x = F.log_softmax(x, dim=1)
        if mask is not None:
            loss = self.criterion(x[mask], target[mask])
        else:
            loss = self.criterion(x, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
