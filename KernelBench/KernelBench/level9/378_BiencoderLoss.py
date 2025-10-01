import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F


class BiencoderLoss(nn.Module):

    def __init__(self):
        super(BiencoderLoss, self).__init__()

    def forward(self, q_vectors: 'T', p_vectors: 'T'):
        score_matrix = torch.mm(q_vectors, torch.transpose(p_vectors, 0, 1))
        score_softmax = F.softmax(score_matrix, dim=1)
        scores = torch.diag(score_softmax)
        loss = torch.mean(torch.neg(torch.log(scores)))
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
