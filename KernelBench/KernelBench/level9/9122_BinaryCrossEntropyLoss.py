import torch
import torch.nn as nn


class BinaryCrossEntropyLoss(nn.Module):
    """(`BinaryCrossEntropyLoss <https://pytorch.org/docs/master/generated/torch.nn.BCELoss.html#bceloss>`__).

    Attributes:
        loss_fct (BCELoss): Binary cross entropy loss function from torch library.
    """

    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss_fct = nn.BCELoss()

    def forward(self, prediction, target):
        return self.loss_fct(prediction, target.float())


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
