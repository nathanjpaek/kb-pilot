import torch
from torch import Tensor
from typing import Optional
from torch import nn


class BWCEWLoss(nn.Module):
    """ Binary weighted cross entropy loss. """

    def __init__(self, positive_class_weight: 'Optional[Tensor]'=None,
        robust_lambda: 'int'=0, confidence_penalty: 'int'=0, **kwargs):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=
            positive_class_weight, **kwargs)
        self.robust_lambda = robust_lambda
        self.confidence_penalty = confidence_penalty

    def forward(self, preds: 'torch.Tensor', target: 'torch.Tensor'):
        train_loss = self.loss_fn(preds, target.float())
        if self.robust_lambda > 0:
            train_loss = (1 - self.robust_lambda
                ) * train_loss + self.robust_lambda / 2
        train_mean_loss = torch.mean(train_loss)
        if self.confidence_penalty > 0:
            probabilities = torch.sigmoid(preds)
            mean_penalty = utils.mean_confidence_penalty(probabilities, 2)
            train_mean_loss += self.confidence_penalty * mean_penalty
        return train_mean_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
