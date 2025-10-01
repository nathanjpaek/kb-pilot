import torch
from torch import nn


class Accuracy(nn.Module):
    label = 'Accuracy'

    def forward(self, prediction, truth):
        prediction = prediction.argmax(dim=1)
        correct = prediction == truth
        accuracy = correct.float().mean()
        return accuracy


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
