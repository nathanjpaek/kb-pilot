import torch
from sklearn.metrics import *
import torch.nn as nn


def accuracy(logits, labels, ignore_index: 'int'=-100):
    with torch.no_grad():
        valid_mask = labels != ignore_index
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


class Accuracy(nn.Module):

    def __init__(self, ignore_index: 'int'=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target, self.ignore_index)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
