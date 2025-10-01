import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._utils
from itertools import product as product
import torch.utils.data.distributed


class Logits(nn.Module):
    """
	Do Deep Nets Really Need to be Deep?
	http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
	"""

    def __init__(self):
        super(Logits, self).__init__()

    def forward(self, out_s, out_t):
        loss = F.mse_loss(out_s, out_t)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
