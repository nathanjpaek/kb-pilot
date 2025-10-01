import torch
from torch import Tensor
from torch import nn
from typing import Union


class DistillationLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, alpha: 'float'=0.95, temp: 'Union[float, int]'=6
        ) ->None:
        super().__init__()
        self.alpha = alpha
        self.temp = temp
        self.kd_loss = nn.KLDivLoss()
        self.entropy_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred_student: 'Tensor', pred_teacher: 'Tensor',
        target: 'Tensor') ->Tensor:
        loss = self.kd_loss(self.log_softmax(pred_student / self.temp),
            self.softmax(pred_teacher / self.temp)) * (self.alpha * self.
            temp * self.temp)
        loss += self.entropy_loss(pred_student, target) * (1.0 - self.alpha)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
