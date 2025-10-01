import torch
from torch import nn
import torch.utils.data.distributed


class Correct(nn.Module):

    def forward(self, classifier, target):
        return classifier.max(dim=1)[1] == target


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
