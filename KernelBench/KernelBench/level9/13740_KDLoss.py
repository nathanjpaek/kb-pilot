import torch
import torch.nn.functional as F
import torch.nn as nn


class KDLoss(nn.Module):
    """Knowledge Distillation Loss"""

    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        loss = F.kl_div(s, t, size_average=False
            ) * self.t ** 2 / stu_pred.shape[0]
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'T': 4}]
