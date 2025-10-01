import torch
from torch import nn
from torch.nn import functional as F


class softCrossEntropy(nn.Module):

    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, _class_num = target.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, target), 1)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
