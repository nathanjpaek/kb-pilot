import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._utils
from itertools import product as product
import torch.utils.data.distributed


class SP(nn.Module):
    """
	Similarity-Preserving Knowledge Distillation
	https://arxiv.org/pdf/1907.09682.pdf
	"""

    def __init__(self):
        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)
        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)
        loss = F.mse_loss(norm_G_s, norm_G_t)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
