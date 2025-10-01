import torch
import torch.nn as nn
import torch.nn.functional as F


class EntMinLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, f_x):
        soft_f_x = F.softmax(f_x, dim=-1)
        log_soft_f_x = F.log_softmax(f_x, dim=-1)
        ent = -torch.sum(soft_f_x * log_soft_f_x) / f_x.shape[0]
        return ent


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
