import torch
import torch.nn as nn


class CpuEmbedding(nn.Module):

    def __init__(self, num_embeddings, embed_dim):
        super(CpuEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.zeros((num_embeddings, embed_dim)))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, embedding_dim)
        """
        return self.weight[x]


def get_inputs():
    return [torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'num_embeddings': 4, 'embed_dim': 4}]
