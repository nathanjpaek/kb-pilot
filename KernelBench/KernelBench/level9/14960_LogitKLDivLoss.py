import torch
from torch.nn import functional as F
from torch import nn
from torchvision import models as models
import torch.onnx
import torch.nn


class LogitKLDivLoss(nn.Module):
    """Kullback–Leibler divergence loss. Inputs predicted and ground truth logits.

    Args:
        T (float): Softmax temperature.
    """

    def __init__(self, T=1):
        super().__init__()
        self.T = T

    def forward(self, p_logits, q_logits, **kwargs):
        log_p = F.log_softmax(p_logits / self.T, dim=1)
        q = F.softmax(q_logits / self.T, dim=1)
        return F.kl_div(log_p, q, reduction='batchmean') * self.T ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
