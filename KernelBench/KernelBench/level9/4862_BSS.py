import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._utils
from itertools import product as product
import torch.utils.data.distributed


class BSS(nn.Module):
    """
	Knowledge Distillation with Adversarial Samples Supporting Decision Boundary
	https://arxiv.org/pdf/1805.05532.pdf
	"""

    def __init__(self, T):
        super(BSS, self).__init__()
        self.T = T

    def forward(self, attacked_out_s, attacked_out_t):
        loss = F.kl_div(F.log_softmax(attacked_out_s / self.T, dim=1), F.
            softmax(attacked_out_t / self.T, dim=1), reduction='batchmean')
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'T': 4}]
