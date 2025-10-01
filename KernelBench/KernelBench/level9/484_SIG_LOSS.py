import torch
from torch import nn


class SIG_LOSS(nn.Module):

    def __init__(self, device):
        super(SIG_LOSS, self).__init__()
        self.m_device = device
        self.m_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, preds, targets):
        loss = self.m_criterion(preds, targets)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0}]
