import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Implementation of Focal Loss.

    Focal loss was proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
    """

    def __init__(self, gamma=0, eps=1e-07, reduction='mean'):
        """Constructor Method for FocalLoss class.

        Args:
            gamma : The focal parameter. Defaults to 0.
            eps : Constant for computational stability.
            reduction: The reduction parameter for Cross Entropy Loss.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: 'torch.Tensor', targets: 'torch.Tensor'
        ) ->torch.Tensor:
        """Forward method.

        Args:
            logits: The raw logits from the network of shape (N,C,*) where C = number of classes , * = extra dims
            targets: The targets of shape (N , *).

        Returns:
            The computed loss value
        """
        logp = self.ce(logits, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean() if self.reduction == 'mean' else loss.sum(
            ) if self.reduction == 'sum' else loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
