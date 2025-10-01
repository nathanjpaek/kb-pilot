import math
import torch
from torch.nn.modules.loss import _Loss
import torch.optim
import torch.nn


class BCE_LOSS(_Loss):

    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        C = input.size(1)
        label = label.reshape(one_hot.shape[0], 1)
        one_hot.scatter_(1, label, 1)
        loss = self.bce_loss(input - math.log(C), one_hot) * C
        return loss


def get_inputs():
    return [torch.rand([4, 2]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
