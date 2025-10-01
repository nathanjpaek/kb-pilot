import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F


class KL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(KL, self).__init__()
        self.T = T

    def forward(self, y_s, p_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False
            ) * self.T ** 2 / p_s.shape[0]
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'T': 4}]
