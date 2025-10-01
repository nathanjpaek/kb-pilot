import torch
import torch.nn as nn


class OcclusionAwareSimilarity(nn.Module):

    def __init__(self, threshold):
        super(OcclusionAwareSimilarity, self).__init__()
        self.threshold = threshold

    def forward(self, similarity_matrix):
        indicator_zero = similarity_matrix <= self.threshold
        similarity_matrix[indicator_zero] = 0
        return similarity_matrix


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'threshold': 4}]
