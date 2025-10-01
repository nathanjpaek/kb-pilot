import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim.lr_scheduler import *
from torchvision.models import *
from torchvision.transforms import *


class CrossEntropyLossWithAuxiliary(nn.CrossEntropyLoss):
    """Cross-entropy loss that can add auxiliary loss if present."""

    def forward(self, input, target):
        """Return cross-entropy loss and add auxiliary loss if possible."""
        if isinstance(input, dict):
            loss = super().forward(input['out'], target)
            if 'aux' in input:
                loss += 0.5 * super().forward(input['aux'], target)
        else:
            loss = super().forward(input, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
