import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseCrossEntropy(nn.Module):

    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()
        logprobs = F.log_softmax(logits, dim=-1)
        loss = -labels * logprobs
        loss = loss.sum(-1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
