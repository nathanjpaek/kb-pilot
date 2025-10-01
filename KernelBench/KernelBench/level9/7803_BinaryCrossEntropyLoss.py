import torch
import torch.nn as nn


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, pos_weight=None, reduction='mean'):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.BCE_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
            reduction=reduction)

    def forward(self, inputs, targets):
        return self.BCE_loss(inputs, targets)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
