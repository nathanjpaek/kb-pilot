import torch
from typing import *
from torch import nn
import torch.nn.functional as F
from torch import functional as F
from torch.nn import functional as F


class CircleLoss(nn.Module):
    """CircleLoss from
	`"Circle Loss: A Unified Perspective of Pair Similarity Optimization"
	<https://arxiv.org/pdf/2002.10857>`_ paper.

	Parameters
	----------
	m: float.
		Margin parameter for loss.
	gamma: int.
		Scale parameter for loss.

	Outputs:
		- **loss**: scalar.
	"""

    def __init__(self, m, gamma):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.dp = 1 - m
        self.dn = m

    def forward(self, x, target):
        similarity_matrix = x @ x.T
        label_matrix = target.unsqueeze(1) == target.unsqueeze(0)
        negative_matrix = label_matrix.logical_not()
        positive_matrix = label_matrix.fill_diagonal_(False)
        sp = torch.where(positive_matrix, similarity_matrix, torch.
            zeros_like(similarity_matrix))
        sn = torch.where(negative_matrix, similarity_matrix, torch.
            zeros_like(similarity_matrix))
        ap = torch.clamp_min(1 + self.m - sp.detach(), min=0.0)
        an = torch.clamp_min(sn.detach() + self.m, min=0.0)
        logit_p = -self.gamma * ap * (sp - self.dp)
        logit_n = self.gamma * an * (sn - self.dn)
        logit_p = torch.where(positive_matrix, logit_p, torch.zeros_like(
            logit_p))
        logit_n = torch.where(negative_matrix, logit_n, torch.zeros_like(
            logit_n))
        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp
            (logit_n, dim=1)).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'m': 4, 'gamma': 4}]
