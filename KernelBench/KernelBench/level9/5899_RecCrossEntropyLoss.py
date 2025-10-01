import torch
from torch import nn


class RecCrossEntropyLoss(nn.Module):

    def __init__(self, rec_ratio):
        super(RecCrossEntropyLoss, self).__init__()
        self.rec_ratio = rec_ratio

    def forward(self, rec, inputs, logits, targets):
        rec_loss = nn.MSELoss()
        cls_loss = nn.CrossEntropyLoss()
        return cls_loss(logits, targets) + self.rec_ratio * rec_loss(rec,
            inputs)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'rec_ratio': 4}]
