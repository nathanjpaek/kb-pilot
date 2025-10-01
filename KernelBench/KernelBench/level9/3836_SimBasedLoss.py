import torch
from torch import nn
import torch.nn.functional as F


class SimBasedLoss(nn.Module):

    def __init__(self):
        super(SimBasedLoss, self).__init__()

    def forward(self, y_s, y_t):
        y_s = F.normalize(y_s, p=2, dim=1)
        y_t = F.normalize(y_t, p=2, dim=1)
        student_sims = torch.matmul(y_s, y_s.T)
        teacher_sims = torch.matmul(y_t, y_t.T)
        loss = F.mse_loss(student_sims, teacher_sims)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
