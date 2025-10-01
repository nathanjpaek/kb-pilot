import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    cross entropy loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction='none')


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
