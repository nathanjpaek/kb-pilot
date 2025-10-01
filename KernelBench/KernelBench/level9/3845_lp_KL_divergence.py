import torch
from torch.utils.data import *
import torch.nn as nn


class lp_KL_divergence(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.normalize = nn.Softmax(dim=-1)

    def forward(self, x, y):
        embed_dim = x.shape[-1]
        x = x.view(-1, embed_dim)
        y = y.view(-1, embed_dim)
        x = self.normalize(x)
        y = self.normalize(y)
        loss = self.loss(x, y)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
