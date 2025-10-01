import torch
from typing import *
from torch import nn
import torch.nn.functional as F
from torch import functional as F
from torch.nn import functional as F


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        return self.temperature ** 2 * torch.mean(torch.sum(-F.softmax(
            teacher_output / self.temperature) * F.log_softmax(
            student_output / self.temperature), dim=1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
