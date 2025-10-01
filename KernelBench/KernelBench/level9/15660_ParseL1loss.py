import torch
from torch import nn
import torch.nn.functional as F


class ParseL1loss(nn.Module):

    def __init__(self):
        super(ParseL1loss, self).__init__()

    def forward(self, output, target, mask):
        mask = (mask == 1).float()
        loss = F.l1_loss(output * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 0.0001)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
