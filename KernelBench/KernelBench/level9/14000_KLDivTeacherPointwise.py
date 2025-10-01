import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *


class KLDivTeacherPointwise(nn.Module):

    def __init__(self):
        super(KLDivTeacherPointwise, self).__init__()
        self.kl = torch.nn.KLDivLoss()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        """
        loss = self.kl(scores_pos, label_pos) + self.kl(scores_neg, label_neg)
        return loss / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
