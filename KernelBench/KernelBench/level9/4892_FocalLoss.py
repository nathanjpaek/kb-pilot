import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implementation of the binary focal loss proposed in:
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: 'float'=1.0, gamma: 'float'=2.0, reduce:
        'str'='mean') ->None:
        """
        Constructor method
        :param alpha: (float) Alpha constant (see paper)
        :param gamma: (float) Gamma constant (ses paper)
        :param reduce: (str) Reduction operation (mean, sum or none)
        """
        super(FocalLoss, self).__init__()
        assert reduce in ['mean', 'sum', 'none'
            ], 'Illegal value of reduce parameter. Use mean, sum or none.'
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, prediction: 'torch.Tensor', label: 'torch.Tensor'
        ) ->torch.Tensor:
        """
        Forward method calculates the dice loss
        :param prediction: (torch.tensor) Prediction tensor including probabilities
        :param label: (torch.tensor) Label tensor (one-hot encoded)
        :return: (torch.tensor) Dice loss
        """
        cross_entropy_loss = F.binary_cross_entropy(prediction, label,
            reduction='none')
        focal_loss = self.alpha * (1.0 - prediction
            ) ** self.gamma * cross_entropy_loss
        if self.reduce == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduce == 'sum':
            focal_loss = torch.sum(focal_loss)
        return focal_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
