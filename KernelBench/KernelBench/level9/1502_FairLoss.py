import torch
from torch import nn


class FairLoss(nn.Module):

    def __init__(self, lamda):
        super(FairLoss, self).__init__()
        self.lamda = lamda

    def forward(self, rep):
        logits = torch.mm(rep, torch.transpose(rep, 0, 1))
        logits = logits - torch.diag_embed(torch.diag(logits))
        logits = logits.abs().sum()
        return logits * self.lamda


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'lamda': 4}]
