import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F


class GKDLoss(nn.Module):
    """Knowledge Distillation Loss"""

    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, stu_pred, tea_pred, label):
        stu_pred_log_softmax = F.log_softmax(stu_pred / self.t, dim=1)
        tea_pred_softmax = F.softmax(tea_pred / self.t, dim=1)
        tea_pred_argmax = torch.argmax(tea_pred_softmax, dim=1)
        mask = torch.eq(label, tea_pred_argmax).float()
        count = mask[mask == 1].size(0)
        mask = mask.unsqueeze(-1)
        only_correct_sample_stu_pred_log_softmax = stu_pred_log_softmax.mul(
            mask)
        only_correct_sample_tea_pred_softmax = tea_pred_softmax.mul(mask)
        only_correct_sample_tea_pred_softmax[
            only_correct_sample_tea_pred_softmax == 0.0] = 1.0
        loss = F.kl_div(only_correct_sample_stu_pred_log_softmax,
            only_correct_sample_tea_pred_softmax, reduction='sum'
            ) * self.t ** 2 / count
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'T': 4}]
