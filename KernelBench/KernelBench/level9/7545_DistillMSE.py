import torch
from torch import nn
import torch.nn.functional as F


class DistillMSE(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self):
        super(DistillMSE, self).__init__()
        pass

    def forward(self, y_s, y_t):
        loss = nn.MSELoss(reduction='mean')(F.softmax(y_s, dim=1), F.
            softmax(y_t, dim=1))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
