import torch
import torch.nn as nn
import torch.utils.data


class CrossEntropyLossIIT(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, preds, labels):
        return self.loss(preds[0], labels[:, 0]) + self.loss(preds[1],
            labels[:, 1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
