import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCrossEntropy(nn.Module):

    def __init__(self):
        super(AttentionCrossEntropy, self).__init__()

    def forward(self, input, target):
        cross_loss = torch.mul(target.float(), F.log_softmax(input, dim=1))
        loss = torch.neg(torch.mean(cross_loss))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
