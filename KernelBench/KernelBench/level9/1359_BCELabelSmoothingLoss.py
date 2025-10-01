import torch
from torch import nn
import torch.nn.functional as F


class BCELabelSmoothingLoss(nn.Module):
    """
    Binary Cross Entropy Loss with label smoothing, takes logits
    """

    def __init__(self, smoothing):
        """
        `smoothing` is the smoothing factor. How much less confident than 100%
         are we on true labels?
        """
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        """ expects target to be binary, and input to be logits
        """
        with torch.no_grad():
            target = torch.abs(target - self.smoothing)
        return F.binary_cross_entropy_with_logits(input, target)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'smoothing': 4}]
