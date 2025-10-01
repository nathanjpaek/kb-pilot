import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):

    def __init__(self, is_weight=False, weight=[]):
        super(CrossEntropy, self).__init__()
        self.is_weight = is_weight
        self.weight = weight

    def forward(self, input, target, batchsize=2):
        target = torch.argmax(target, dim=1)
        if self.is_weight is True:
            loss = F.cross_entropy(input, target, torch.tensor(self.weight)
                .float())
        else:
            loss = F.cross_entropy(input, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
