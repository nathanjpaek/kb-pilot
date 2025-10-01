import torch
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *


class CrossEntropyLossWithSoftLabel(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        log_probs = self.logsoftmax(input)
        loss = (-target * log_probs).sum(dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
