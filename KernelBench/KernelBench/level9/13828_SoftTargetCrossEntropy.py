import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.optim
import torch._utils
import torch.nn


class SoftTargetCrossEntropy(_Loss):

    def __init__(self, loss_weight=1.0):
        super(SoftTargetCrossEntropy, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, label) ->torch.Tensor:
        loss = torch.sum(-label * F.log_softmax(input, dim=-1), dim=-1)
        return loss.mean() * self.loss_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
