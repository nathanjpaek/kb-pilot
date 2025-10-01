import torch
import torch.nn as nn


class selfCrossEntropy(nn.Module):

    def __init__(self):
        super(selfCrossEntropy, self).__init__()

    def forward(self, output, target):
        output = nn.functional.softmax(output, dim=0)
        return -torch.mean(torch.sum(target * torch.log(torch.clamp(output,
            min=1e-10)) + (1 - target) * torch.log(torch.clamp(1 - output,
            min=1e-10)), 1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
