import torch
from torch import Tensor
from typing import List
from typing import Optional
from typing import Union
from torch import nn


class LogitsInputsMixin:

    @classmethod
    def get_loss_inputs(cls):
        """Maps loss to the desired predicted input type."""
        return LOGITS


class SigmoidCrossEntropyLoss(nn.Module, LogitsInputsMixin):

    def __init__(self, class_weights: 'Optional[Union[Tensor, List]]'=None,
        **kwargs):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if class_weights:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none',
                pos_weight=torch.Tensor(class_weights))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds: 'Tensor', target: 'Tensor') ->Tensor:
        if preds.ndim != 2:
            raise RuntimeError(
                'SigmoidCrossEntropyLoss currently supported for 2D tensors.')
        element_loss = self.loss_fn(preds.type(torch.float32), target.type(
            torch.float32))
        loss = torch.sum(element_loss, dim=1)
        loss = torch.mean(loss)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
