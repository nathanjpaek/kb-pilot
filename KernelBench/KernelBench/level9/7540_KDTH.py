import torch
from torch import nn
import torch.nn.functional as F


class KDTH(nn.Module):
    """KD with a Teacher Head auxiliary loss"""

    def __init__(self, T=4):
        super(KDTH, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        y_s_th = y_s[1]
        y_s = y_s[0]
        p_t = F.softmax(y_t / self.T, dim=1)
        p_s_th = F.log_softmax(y_s_th / self.T, dim=1)
        loss_th = F.kl_div(p_s_th, p_t, size_average=False
            ) * self.T ** 2 / y_s.shape[0]
        return loss_th


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
