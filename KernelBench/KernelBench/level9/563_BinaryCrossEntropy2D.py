import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss


class BinaryCrossEntropy2D(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self):
        super(BinaryCrossEntropy2D, self).__init__()
        self.nll_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        """
        Forward pass
        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(inputs, targets)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
