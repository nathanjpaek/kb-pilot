import torch
from torch import Tensor
from torch.functional import Tensor
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """
    Originally from:
        https://arxiv.org/pdf/1409.0473v5.pdf

    Also referenced to as Content Based Attention:
        https://arxiv.org/pdf/1506.03134v1.pdf

    Attention is learned for a query vector over a set of vectors. 
    If we have a query vector and then a key matrix with the size (n,k), 
    we will create attention vector over n

    
    NOTE!
    We mask out the attention scores for the positions which its impossible for 
    the segments to attend to. I.e. the padding.


    """

    def __init__(self, input_dim: 'int'):
        super().__init__()
        self.W1 = nn.Linear(input_dim, input_dim)
        self.W2 = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1)

    def forward(self, query: 'Tensor', key: 'Tensor', mask: 'Tensor',
        softmax=False):
        key_out = self.W1(key)
        query_out = self.W2(query.unsqueeze(1))
        ui = self.v(torch.tanh(key_out + query_out)).squeeze(-1)
        mask = mask.type(torch.bool)
        ui[~mask] = float('-inf')
        if softmax:
            ui = torch.softmax(ui)
        return ui


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
