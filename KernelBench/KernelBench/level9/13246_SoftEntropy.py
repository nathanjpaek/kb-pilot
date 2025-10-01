import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import *
from torch.optim.lr_scheduler import *


class SoftEntropy(nn.Module):

    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        loss = (-F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
