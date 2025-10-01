import torch
import torch.nn.functional as F


class ConditionalEntropyLoss(torch.nn.Module):

    def __init__(self, model):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x, weight):
        loss = F.softmax(x, dim=1) * F.log_softmax(x, dim=1) * weight
        loss = loss.sum(dim=1)
        return -1.0 * loss.mean(dim=0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'model': 4}]
