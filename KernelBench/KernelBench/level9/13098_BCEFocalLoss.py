import torch
from torch import nn


class BCEFocalLoss(nn.Module):
    """Implementation of Focal Loss for Binary Classification Problems.

    Focal loss was proposed in `Focal Loss for Dense Object Detection_.
    <https://arxiv.org/abs/1708.02002>`_.
    """

    def __init__(self, gamma=0, eps=1e-07, reduction='mean'):
        """Constructor Method for FocalLoss class.

        Args:
            gamma : The focal parameter. Defaults to 0.
            eps : Constant for computational stability.
            reduction: The reduction parameter for Cross Entropy Loss.
        """
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: 'torch.Tensor', targets: 'torch.Tensor'
        ) ->torch.Tensor:
        """Forward method.

        Args:
            logits: The raw logits from the network of shape (N,k)                     where C = number of classes , k = extra dims
            targets: The targets

        Returns:
            The computed loss value
        """
        targets = targets.view(logits.shape)
        logp = self.bce(logits, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean() if self.reduction == 'mean' else loss.sum(
            ) if self.reduction == 'sum' else loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
