import torch
from typing import Optional
from typing import List
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


class WeightedCE(nn.Module):
    """Mask weighted multi-class cross-entropy (CE) loss.
    """

    def __init__(self, class_weight: 'Optional[List[float]]'=None):
        super().__init__()
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight)

    def forward(self, pred, target, weight_mask=None):
        if self.class_weight is not None:
            self.class_weight = self.class_weight
        loss = F.cross_entropy(pred, target, weight=self.class_weight,
            reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
