import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection [https://arxiv.org/abs/1708.02002]
    Digest the paper as below:
        α, balances the importance of positive/negative examples
        γ, focusing parameter that controls the strength of the modulating term

            CE(pt) = −log(pt) ==> pt = exp(-CE)
            FL(pt) = −α((1 − pt)^γ) * log(pt)

        In general α should be decreased slightly as γ is increased (for γ = 2, α = 0.25 works best).
    """

    def __init__(self, focusing_param=2, balance_param=0.25):
        super().__init__()
        self.gamma = focusing_param
        self.alpha = balance_param

    def forward(self, inputs, targets, weights=None):
        logpt = -binary_cross_entropy(inputs, targets, weights)
        pt = torch.exp(logpt)
        focal_loss = -(1 - pt) ** self.gamma * logpt
        balanced_focal_loss = self.alpha * focal_loss
        return balanced_focal_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
