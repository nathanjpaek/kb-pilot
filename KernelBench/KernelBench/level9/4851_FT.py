import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._utils
from itertools import product as product
import torch.utils.data.distributed


class FT(nn.Module):
    """
	araphrasing Complex Network: Network Compression via Factor Transfer
	http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf
	"""

    def __init__(self):
        super(FT, self).__init__()

    def forward(self, factor_s, factor_t):
        loss = F.l1_loss(self.normalize(factor_s), self.normalize(factor_t))
        return loss

    def normalize(self, factor):
        norm_factor = F.normalize(factor.view(factor.size(0), -1))
        return norm_factor


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
