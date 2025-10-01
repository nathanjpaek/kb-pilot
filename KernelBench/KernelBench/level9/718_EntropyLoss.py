import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F


class EntropyLoss(nn.Module):

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, input):
        prob = F.softmax(input, dim=1)
        if (prob < 0).any() or (prob > 1).any():
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')
        prob = prob + 1e-16
        H = torch.mean(torch.sum(prob * torch.log(prob), dim=1))
        return H


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
