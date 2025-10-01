import torch
from torch import nn
import torch.nn.functional as F


class SoftCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, target):
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, _class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
