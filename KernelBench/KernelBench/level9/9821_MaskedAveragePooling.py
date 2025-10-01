import torch
from torch import nn


class MaskedAveragePooling(nn.Module):

    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix):
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix.sum(dim=-1) != 0).sum(dim=1,
            keepdim=True)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 
            1e-12)
        return embedding_vec


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
