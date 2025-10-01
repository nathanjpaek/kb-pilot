import torch
import warnings
import torch.nn as nn
from torchvision.transforms import *


class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """

    def __init__(self):
        super(RingLoss, self).__init__()
        warnings.warn('This method is deprecated')
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(self, x):
        loss = ((x.norm(p=2, dim=1) - self.radius) ** 2).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
