import torch
import torch.nn.functional as F
from torch import nn


class Weighed_Bce_Loss(nn.Module):

    def __init__(self):
        super(Weighed_Bce_Loss, self).__init__()

    def forward(self, x, label):
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        label = label.view(-1, 1, label.shape[1], label.shape[2])
        label_t = (label == 1).float()
        label_f = (label == 0).float()
        p = torch.sum(label_t) / (torch.sum(label_t) + torch.sum(label_f))
        w = torch.zeros_like(label)
        w[label == 1] = p
        w[label == 0] = 1 - p
        loss = F.binary_cross_entropy(x, label, weight=w)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
