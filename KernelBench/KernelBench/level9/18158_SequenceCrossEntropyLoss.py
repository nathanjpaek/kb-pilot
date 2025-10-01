import torch
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import *
from sklearn.metrics import *


class SequenceCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, inputs, targets):
        losses = self.criterion(inputs, targets)
        return losses.sum() / inputs.shape[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
