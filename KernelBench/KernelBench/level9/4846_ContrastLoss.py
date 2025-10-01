import torch
import torch.nn as nn
import torch._utils
from itertools import product as product
import torch.utils.data.distributed


class ContrastLoss(nn.Module):
    """
	contrastive loss, corresponding to Eq.(18)
	"""

    def __init__(self, n_data, eps=1e-07):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data
        self.eps = eps

    def forward(self, x):
        bs = x.size(0)
        N = x.size(1) - 1
        M = float(self.n_data)
        pos_pair = x.select(1, 0)
        log_pos = torch.div(pos_pair, pos_pair.add(N / M + self.eps)).log_()
        neg_pair = x.narrow(1, 1, N)
        log_neg = torch.div(neg_pair.clone().fill_(N / M), neg_pair.add(N /
            M + self.eps)).log_()
        loss = -(log_pos.sum() + log_neg.sum()) / bs
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_data': 4}]
