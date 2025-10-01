import torch
import torch.nn as nn
import torch.nn.functional as F


class CLOSS(nn.Module):

    def __init__(self, m=1.0):
        super().__init__()
        self.m = m

    def forward(self, pp_pair, pn_pair):
        basic_loss = F.sigmoid(pp_pair) - F.sigmoid(pn_pair) + self.m
        loss = torch.max(torch.zeros_like(basic_loss), basic_loss).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
