import torch
import torch.nn as nn
import torch._utils
from itertools import product as product
import torch.utils.data.distributed


class AB(nn.Module):
    """
	Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
	https://arxiv.org/pdf/1811.03233.pdf
	"""

    def __init__(self, margin):
        super(AB, self).__init__()
        self.margin = margin

    def forward(self, fm_s, fm_t):
        loss = (fm_s + self.margin).pow(2) * ((fm_s > -self.margin) & (fm_t <=
            0)).float() + (fm_s - self.margin).pow(2) * ((fm_s <= self.
            margin) & (fm_t > 0)).float()
        loss = loss.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
