import math
import torch
from torch.nn.modules.loss import _Loss
import torch.optim
import torch._utils
import torch.nn


class BCE_LOSS(_Loss):

    def __init__(self, loss_weight=1.0, bias=False):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.loss_weight = loss_weight
        self.bias = bias

    def forward(self, input, label):
        C = input.size(1) if self.bias else 1
        if label.dim() == 1:
            one_hot = torch.zeros_like(input)
            label = label.reshape(one_hot.shape[0], 1)
            one_hot.scatter_(1, label, 1)
            loss = self.bce_loss(input, one_hot)
        elif label.dim() > 1:
            loss = self.bce_loss(input - math.log(C), label) * C
        return loss.mean() * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
