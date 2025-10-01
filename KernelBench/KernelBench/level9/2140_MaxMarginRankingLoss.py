import torch
import numpy as np
import torch as th
import torch.nn.functional as F


class MaxMarginRankingLoss(th.nn.Module):

    def __init__(self, margin=1.0, negative_weighting=False, batch_size=1,
        n_pair=1, hard_negative_rate=0.5):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 -
                easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = th.tensor(mm_mask) * (batch_size * (1 -
                easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = th.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + F.relu(self.
            margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1:
            max_margin = max_margin * self.mm_mask
        return max_margin.mean()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
