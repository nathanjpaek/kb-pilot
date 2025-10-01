import torch
import torch.nn as nn


class PairwiseLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.m = 0

    def forward(self, pos_out, neg_out):
        distance = 1 - pos_out + neg_out
        loss = torch.mean(torch.max(torch.tensor(0), distance))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
