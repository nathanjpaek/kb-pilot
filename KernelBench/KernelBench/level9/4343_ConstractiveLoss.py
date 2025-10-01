import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class ConstractiveLoss(nn.Module):

    def __init__(self, margin=2.0, dist_flag='l2'):
        super(ConstractiveLoss, self).__init__()
        self.margin = margin
        self.dist_flag = dist_flag

    def various_distance(self, out_vec_t0, out_vec_t1):
        if self.dist_flag == 'l2':
            distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
        if self.dist_flag == 'l1':
            distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
        if self.dist_flag == 'cos':
            similarity = F.cosine_similarity(out_vec_t0, out_vec_t1)
            distance = 1 - 2 * similarity / np.pi
        return distance

    def forward(self, out_vec_t0, out_vec_t1, label):
        distance = self.various_distance(out_vec_t0, out_vec_t1)
        constractive_loss = torch.sum((1 - label) * torch.pow(distance, 2) +
            label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return constractive_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
