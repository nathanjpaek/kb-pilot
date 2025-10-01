import torch
import torch.utils.data
import torch
import torch.nn as nn
from torch.nn import functional as F


class GramLoss(nn.Module):

    def __init__(self):
        super(GramLoss, self).__init__()

    def forward(self, input, target):
        input = input.reshape(input.shape[0], input.shape[1], -1)
        target = target.reshape(target.shape[0], target.shape[1], -1)
        None
        input_gram = torch.matmul(input, torch.permute(input, (0, 2, 1)))
        None
        target_gram = torch.matmul(target, torch.permute(target, (0, 2, 1)))
        return F.l1_loss(input_gram, target_gram)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
