import torch
import torch.utils.data
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):

    def __init__(self, label_nc):
        super(CrossEntropyLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss2d()

    def forward(self, output, label):
        label = label.long().max(1)[1]
        output = self.softmax(output)
        return self.criterion(output, label)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'label_nc': 4}]
