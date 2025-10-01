import math
import torch
import torch.nn as nn


class LatentAtten(nn.Module):
    """
    Attention on latent representation
    """

    def __init__(self, h_dim, key_dim=None) ->None:
        super(LatentAtten, self).__init__()
        if key_dim is None:
            key_dim = h_dim
        self.key_dim = key_dim
        self.key_layer = nn.Linear(h_dim, key_dim)
        self.query_layer = nn.Linear(h_dim, key_dim)

    def forward(self, h_M, h_R):
        key = self.key_layer(h_M)
        query = self.query_layer(h_R)
        atten = key @ query.transpose(0, 1) / math.sqrt(self.key_dim)
        atten = torch.softmax(atten, 1)
        return atten


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'h_dim': 4}]
