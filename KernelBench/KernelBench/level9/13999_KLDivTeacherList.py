import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *


class KLDivTeacherList(nn.Module):

    def __init__(self):
        super(KLDivTeacherList, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, scores, labels):
        """
        """
        loss = self.kl(scores.softmax(-1), labels.softmax(-1))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
