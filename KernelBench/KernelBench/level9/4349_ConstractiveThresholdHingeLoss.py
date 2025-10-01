import torch
import torch.nn as nn
from torch.nn import functional as F


class ConstractiveThresholdHingeLoss(nn.Module):

    def __init__(self, hingethresh=0.0, margin=2.0):
        super(ConstractiveThresholdHingeLoss, self).__init__()
        self.threshold = hingethresh
        self.margin = margin

    def forward(self, out_vec_t0, out_vec_t1, label):
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
        similar_pair = torch.clamp(distance - self.threshold, min=0.0)
        dissimilar_pair = torch.clamp(self.margin - distance, min=0.0)
        constractive_thresh_loss = torch.sum((1 - label) * torch.pow(
            similar_pair, 2) + label * torch.pow(dissimilar_pair, 2))
        return constractive_thresh_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
