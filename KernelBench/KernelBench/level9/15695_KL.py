import torch
import torch.nn as nn
import torch.nn.functional as F


class KL(nn.Module):

    def __init__(self, reduction='batchmean'):
        super(KL, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1), F.softmax(target, dim
            =-1), reduction='batchmean')
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
