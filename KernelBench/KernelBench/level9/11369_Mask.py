import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from typing import *


class Mask(nn.Module):

    def forward(self, seq, mask):
        seq_mask = torch.unsqueeze(mask, 2)
        seq_mask = torch.transpose(seq_mask.repeat(1, 1, seq.size()[1]), 1, 2)
        return seq.where(torch.eq(seq_mask, 1), torch.zeros_like(seq))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
