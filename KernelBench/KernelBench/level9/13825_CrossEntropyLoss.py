import torch
from torch.nn.modules.loss import _Loss
import torch.optim
import torch._utils
import torch.nn


class CrossEntropyLoss(_Loss):

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, input, label):
        loss = self.ce_loss(input, label)
        return loss * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
