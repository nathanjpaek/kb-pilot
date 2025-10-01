import torch
import torch.nn as nn


class AdjMSELoss(nn.Module):

    def __init__(self):
        super(AdjMSELoss, self).__init__()

    def forward(self, outputs, labels):
        loss = torch.abs(outputs - labels)
        adj_fact = torch.mean(torch.abs(labels)) ** 2
        adj = torch.exp(-outputs * labels / adj_fact)
        loss = loss * adj
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
