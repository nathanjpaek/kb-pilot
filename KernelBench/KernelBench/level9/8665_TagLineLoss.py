import torch
from torch import nn


class TagLineLoss(nn.Module):

    def __init__(self):
        super(TagLineLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(input=output, target=target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
